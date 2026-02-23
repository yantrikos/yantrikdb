use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use aidb_core::AIDB;

fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

fn seed_db(db: &AIDB, n: usize, dim: usize) {
    let meta = serde_json::json!({});
    for i in 0..n {
        let emb = vec_seed(i as f32 * 0.37, dim);
        db.record(
            &format!("Memory number {} about topic {}", i, i % 10),
            if i % 2 == 0 { "episodic" } else { "semantic" },
            0.3 + (i % 7) as f64 * 0.1,
            (i % 5) as f64 * 0.2 - 0.4,
            604800.0,
            &meta,
            &emb,
        )
        .unwrap();
    }
}

fn bench_record(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    let meta = serde_json::json!({});
    let mut i = 0u64;

    c.bench_function("record", |b| {
        b.iter(|| {
            let emb = vec_seed(i as f32 * 0.37 + 10000.0, dim);
            db.record(
                black_box(&format!("bench record {}", i)),
                "episodic",
                0.5,
                0.0,
                604800.0,
                &meta,
                &emb,
            )
            .unwrap();
            i += 1;
        })
    });
}

fn bench_recall(c: &mut Criterion) {
    let dim = 64;
    let mut group = c.benchmark_group("recall");

    for &n in &[100, 500, 1000] {
        let db = AIDB::new(":memory:", dim).unwrap();
        seed_db(&db, n, dim);
        let query = vec_seed(999.0, dim);

        group.bench_with_input(BenchmarkId::new("top10", n), &n, |b, _| {
            b.iter(|| db.recall(black_box(&query), 10, None, None, false).unwrap())
        });
    }
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    let meta = serde_json::json!({});
    let rid = db
        .record("lookup target", "episodic", 0.5, 0.0, 604800.0, &meta, &vec_seed(1.0, dim))
        .unwrap();

    c.bench_function("get", |b| {
        b.iter(|| db.get(black_box(&rid)).unwrap())
    });
}

fn bench_stats(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db(&db, 100, dim);

    c.bench_function("stats_100", |b| {
        b.iter(|| db.stats().unwrap())
    });
}

fn bench_relate(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db(&db, 100, dim);
    let mut i = 0u64;

    c.bench_function("relate", |b| {
        b.iter(|| {
            db.relate(
                &format!("entity_{}", i),
                &format!("entity_{}", i + 1),
                "related_to",
                1.0,
            )
            .unwrap();
            i += 1;
        })
    });
}

fn bench_decay(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db(&db, 100, dim);

    c.bench_function("decay_100", |b| {
        b.iter(|| db.decay(black_box(0.01)).unwrap())
    });
}

fn bench_bulk_insert(c: &mut Criterion) {
    let dim = 64;
    let meta = serde_json::json!({});

    c.bench_function("bulk_insert_500", |b| {
        b.iter(|| {
            let db = AIDB::new(":memory:", dim).unwrap();
            for i in 0..500 {
                let emb = vec_seed(i as f32 * 0.37, dim);
                db.record(
                    &format!("bulk {}", i),
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &meta,
                    &emb,
                )
                .unwrap();
            }
        })
    });
}

criterion_group!(
    benches,
    bench_record,
    bench_get,
    bench_stats,
    bench_relate,
    bench_decay,
    bench_recall,
    bench_bulk_insert,
);
criterion_main!(benches);
