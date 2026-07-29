# wordpress-expert constitution

Injected unconditionally on every request while mounted, so it pays a
token cost each time. It earns that only by carrying rules that change
what gets written. Anything that is merely *true* belongs in the corpus,
where retrieval fetches it when the question calls for it.

Selection rule for what is here: it is a defect a competent model
produces by default, in code that looks correct and passes a smoke test.

## Escape at the point of output, matched to position

Every value entering HTML is escaped where it is printed, never where it
is stored: `esc_html()` in text, `esc_attr()` inside a quoted attribute,
`esc_url()` for URLs, `esc_js()` in inline script, `wp_kses_post()` for
content that may carry markup. Escaping on input corrupts the stored
value and double-escapes on display.

## Sanitize every superglobal, unslashed first

Nothing from `$_POST`, `$_GET`, `$_REQUEST` or `$_COOKIE` is used raw.
`wp_unslash()` first, then the field-appropriate sanitizer —
`sanitize_text_field()`, `sanitize_email()`, `sanitize_key()`,
`absint()`, `wp_kses_post()`. Not a generic cast.

## Every query is prepared; every LIKE is esc_like'd first

SQL through `$wpdb` uses `$wpdb->prepare()` with `%d` / `%s` / `%f` —
never interpolation, even of a value believed safe. Table names come
from `$wpdb->prefix`, never a literal `wp_`. Search terms pass through
`$wpdb->esc_like()` before being wrapped in `%`.

## A nonce is proof of intent, never of permission

State-changing handlers verify the nonce **and** call
`current_user_can()`. Both, always, in that order. A handler with a
nonce and no capability check is the single most common plugin-review
rejection, and it is a real privilege-escalation bug.

## Check capabilities, never role names

`current_user_can('edit_posts')`, never `current_user_can('editor')`.
Roles are site-customisable bundles; checking one silently grants or
denies the wrong people. Object-level checks pass the id:
`current_user_can('edit_post', $post_id)`.

## Every REST route declares a real permission_callback

`register_rest_route` without `permission_callback` is a bug, and
`'__return_true'` is a deliberate decision to publish the endpoint to
the internet — it appears only with a comment saying so. Route `args`
carry `validate_callback` and `sanitize_callback` rather than the
callback re-checking by hand.

## Hook in; never run at file scope

Behaviour attaches via `add_action()` / `add_filter()` at the right
moment: post types and taxonomies on `init`, theme supports on
`after_setup_theme`, cross-plugin work on `plugins_loaded`, assets on
`wp_enqueue_scripts` / `admin_enqueue_scripts`. Core is never modified.
A callback taking more than one argument declares `$accepted_args`.

## Filters return; actions do not

A filter callback returns a value on **every** path, including early
exits and error branches. Returning null or falling off the end of a
`the_title` or `the_content` filter blanks that value across the entire
site, feeds and REST included.

## Assets are enqueued and conditional

`wp_enqueue_script()` / `wp_enqueue_style()` with real dependencies and
a real `$ver` — never an echoed `<script>` tag. Loading is gated to the
screens that need it (`is_singular()`, the `$hook_suffix`), and data
reaches JS via `wp_localize_script()` or `wp_add_inline_script()`, never
PHP interpolated into a script block.

## Prefix every global name

Functions, classes, constants, options, transients, meta keys, script
and style handles, hooks, shortcodes and post types all share one
namespace with every other plugin on the site. All of them carry the
plugin prefix, and post type keys stay within 20 characters.

## Options that are large or rarely read are not autoloaded

`update_option($k, $v, false)` unless the value is small and needed on
most requests. Expensive results go in transients with an explicit
expiry, never in options. `posts_per_page => -1` does not appear;
queries that do not paginate pass `no_found_rows => true`.

## Deactivation clears schedules; only uninstall deletes data

`register_activation_hook` creates schema and flushes rewrite rules
once. `register_deactivation_hook` unschedules events and touches no
user data. Deletion lives in `uninstall.php`. `flush_rewrite_rules()`
never runs on a normal request.

## User-facing strings are translated with a literal text domain

`__()`, `esc_html__()`, `_n()`, `_x()` wrap every user-visible string,
and the text domain is a literal — never a variable or constant, which
the extraction tooling cannot read and silently skips.

## Outbound HTTP uses the WordPress HTTP API with a timeout

`wp_remote_get()` / `wp_remote_post()` with an explicit `timeout`, then
`is_wp_error()` checked before the body is touched. Raw cURL or
`file_get_contents()` on a URL bypasses site proxy and host-blocking
configuration and hangs page loads.

## Failures return WP_Error, not false

Anything a caller must diagnose returns `new WP_Error($code, $message,
['status' => $http_status])`. In REST callbacks that status becomes the
response code, which is how a proper 404 or 409 is produced.
