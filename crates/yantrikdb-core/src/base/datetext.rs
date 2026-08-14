//! Deterministic EVENT-TIME extraction from memory text.
//!
//! # The two clocks
//!
//! A memory has two independent times and the engine has only ever stored one:
//!
//! * **transaction time** — when the memory was written. That is `created_at`,
//!   and it is what every existing query filters on.
//! * **event time** — when the thing the memory *describes* happens. It exists
//!   only inside the prose: "the deployment deadline is March 15, 2024".
//!
//! Storing only the first is actively misleading whenever they disagree. A
//! measured example from a conversation corpus: a record written on
//! 2024-03-14 whose text mentions December 15 2023, December 16 2023,
//! January 15 2024, February 15 2024 and April 15 2024. Its `created_at` is
//! not merely imprecise about those events, it lies outside their entire
//! range. Asked "how many weeks between finishing X and the deadline", nothing
//! in the record's structured fields can answer, because both operands are
//! prose. In that corpus 9.3% of records carried at least one such date, 149
//! distinct dates in total, none of them queryable.
//!
//! # Why deterministic
//!
//! No model call. Extraction runs on the write path of an embedded database,
//! where an LLM round trip per `record()` would be an unacceptable cost and an
//! unacceptable dependency — the engine's whole proposition is that
//! remembering something requires no inference. Regex-free hand parsing keeps
//! it allocation-light and auditable.
//!
//! # What is deliberately NOT parsed
//!
//! * Relative expressions ("next Friday", "in three weeks"). They need a
//!   reference point and a calendar, and resolving them wrongly is worse than
//!   not resolving them: a wrong event time is indistinguishable from a right
//!   one downstream.
//! * Bare numeric forms like `03/04/2024`. `DD/MM` and `MM/DD` cannot be told
//!   apart without a locale, and guessing silently produces off-by-months
//!   errors in a field callers are invited to trust.
//! * Years outside 1900..=2200, which are almost always version numbers,
//!   quantities, or identifiers rather than dates.
//!
//! The bar is: a date this extracts should be one a careful reader would
//! agree is unambiguously a date.

/// A date found in text, as (epoch_seconds_utc_midnight, iso_yyyy_mm_dd).
#[derive(Debug, Clone, PartialEq)]
pub struct EventDate {
    pub epoch: f64,
    pub iso: String,
}

const MONTHS: [(&str, u32); 12] = [
    ("january", 1),
    ("february", 2),
    ("march", 3),
    ("april", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
];

fn month_from(word: &str) -> Option<u32> {
    let w = word.trim_end_matches('.').to_ascii_lowercase();
    if w.len() < 3 {
        return None;
    }
    // Accept full names and the conventional 3+ letter abbreviations ("Sept").
    MONTHS
        .iter()
        .find(|(full, _)| full.starts_with(&w) && w.len() >= 3)
        .map(|(_, n)| *n)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days since the Unix epoch for a proleptic Gregorian Y-M-D.
///
/// Hand-rolled rather than pulled from a date crate: this is the only
/// calendar arithmetic in the engine and it does not justify a dependency
/// (nor the supply-chain surface) on the write path.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    // Howard Hinnant's civil_from_days inverse; shifts the year to start in
    // March so the leap day lands at the end of the cycle.
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = ((m + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn to_event(y: i64, m: u32, d: u32) -> Option<EventDate> {
    if !(1900..=2200).contains(&y) || m == 0 || m > 12 || d == 0 || d > days_in_month(y, m) {
        return None;
    }
    Some(EventDate {
        epoch: (days_from_civil(y, m, d) * 86_400) as f64,
        iso: format!("{y:04}-{m:02}-{d:02}"),
    })
}

fn parse_u(s: &str) -> Option<i64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Every unambiguous calendar date mentioned in `text`, deduplicated and
/// sorted ascending.
///
/// Recognises, case-insensitively:
///   * `March 15, 2024` / `Mar 15 2024` / `15 March 2024`
///   * `2024-03-15` and `March-15-2024`
pub fn extract_event_dates(text: &str) -> Vec<EventDate> {
    let mut found: Vec<EventDate> = Vec::new();
    let bytes = text.as_bytes();

    // ISO `YYYY-MM-DD`, scanned over ASCII digit runs.
    let mut i = 0usize;
    while i + 10 <= bytes.len() {
        let w = &text[i..i + 10];
        let b = w.as_bytes();
        if b[4] == b'-'
            && b[7] == b'-'
            && b[..4].iter().all(u8::is_ascii_digit)
            && b[5..7].iter().all(u8::is_ascii_digit)
            && b[8..10].iter().all(u8::is_ascii_digit)
        {
            // Reject when glued to more digits (a longer number, not a date).
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_digit();
            let after_ok = i + 10 >= bytes.len() || !bytes[i + 10].is_ascii_digit();
            if before_ok && after_ok {
                if let (Some(y), Some(m), Some(d)) =
                    (parse_u(&w[0..4]), parse_u(&w[5..7]), parse_u(&w[8..10]))
                {
                    if let Some(e) = to_event(y, m as u32, d as u32) {
                        found.push(e);
                    }
                }
            }
        }
        i += 1;
    }

    // Word forms. Split on whitespace and the separators that appear between
    // date parts, keeping it allocation-cheap.
    let toks: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == '[' || c == ']')
        .filter(|t| !t.is_empty())
        .collect();
    for w in toks.windows(3) {
        let (a, b, c) = (
            w[0].trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-'),
            w[1].trim_matches(|ch: char| !ch.is_alphanumeric()),
            w[2].trim_matches(|ch: char| !ch.is_alphanumeric()),
        );
        // "March 15, 2024" — the comma is stripped by the trim above.
        if let (Some(m), Some(d), Some(y)) = (month_from(a), parse_u(b), parse_u(c)) {
            if let Some(e) = to_event(y, m, d as u32) {
                found.push(e);
                continue;
            }
        }
        // "15 March 2024"
        if let (Some(d), Some(m), Some(y)) = (parse_u(a), month_from(b), parse_u(c)) {
            if let Some(e) = to_event(y, m, d as u32) {
                found.push(e);
            }
        }
    }
    // Hyphenated `March-15-2024`, which is how some corpora stamp turns.
    for tok in text.split(|c: char| c.is_whitespace() || c == '[' || c == ']' || c == '|') {
        let parts: Vec<&str> = tok
            .trim_matches(|c: char| !c.is_alphanumeric())
            .split('-')
            .collect();
        if parts.len() == 3 {
            if let (Some(m), Some(d), Some(y)) =
                (month_from(parts[0]), parse_u(parts[1]), parse_u(parts[2]))
            {
                if let Some(e) = to_event(y, m, d as u32) {
                    found.push(e);
                }
            }
        }
    }

    found.sort_by(|x, y| x.epoch.total_cmp(&y.epoch).then_with(|| x.iso.cmp(&y.iso)));
    found.dedup_by(|x, y| x.iso == y.iso);
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isos(t: &str) -> Vec<String> {
        extract_event_dates(t).into_iter().map(|e| e.iso).collect()
    }

    #[test]
    fn parses_the_common_written_forms() {
        assert_eq!(isos("due March 15, 2024 sharp"), vec!["2024-03-15"]);
        assert_eq!(isos("due Mar 15 2024"), vec!["2024-03-15"]);
        assert_eq!(isos("due 15 March 2024"), vec!["2024-03-15"]);
        assert_eq!(isos("due 2024-03-15."), vec!["2024-03-15"]);
        assert_eq!(
            isos("[March-15-2024 | Turn 0] User: hi"),
            vec!["2024-03-15"]
        );
    }

    /// The motivating case: one record, five events, none of them its own
    /// write time. Order matters — callers use first/last as a range.
    #[test]
    fn recovers_a_whole_range_from_one_record() {
        let t = "Started December 15, 2023, shipped January 15, 2024, \
                 reviewed February 15, 2024, deadline March 15, 2024.";
        assert_eq!(
            isos(t),
            vec!["2023-12-15", "2024-01-15", "2024-02-15", "2024-03-15"]
        );
    }

    /// A wrong event time is worse than none: it is indistinguishable from a
    /// right one downstream, so ambiguous and relative forms stay unparsed.
    #[test]
    fn refuses_what_it_cannot_know() {
        assert!(isos("let's meet next Friday").is_empty());
        assert!(isos("in three weeks from now").is_empty());
        assert!(
            isos("shipped 03/04/2024").is_empty(),
            "DD/MM vs MM/DD is a guess"
        );
        assert!(isos("version 2024-1 of the spec").is_empty());
    }

    #[test]
    fn rejects_impossible_and_non_dates() {
        assert!(isos("February 30, 2024").is_empty());
        assert!(isos("March 0, 2024").is_empty());
        assert!(isos("the year 1200 BC").is_empty());
        assert!(
            isos("id 20240315123456").is_empty(),
            "digit run, not a date"
        );
        assert_eq!(
            isos("February 29, 2024"),
            vec!["2024-02-29"],
            "leap year is real"
        );
        assert!(isos("February 29, 2023").is_empty(), "not a leap year");
    }

    #[test]
    fn deduplicates_and_sorts() {
        let t = "March 15, 2024 and again March 15, 2024, plus January 2, 2024";
        assert_eq!(isos(t), vec!["2024-01-02", "2024-03-15"]);
    }

    #[test]
    fn epoch_matches_utc_midnight() {
        let e = &extract_event_dates("2024-03-15")[0];
        assert_eq!(e.epoch, 1_710_460_800.0);
        let e = &extract_event_dates("1970-01-01")[0];
        assert_eq!(e.epoch, 0.0);
    }
}
