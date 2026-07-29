# wordpress-expert corpus

Working knowledge for building WordPress plugins and themes that survive
a plugin-review and a production site. One entry per `## ` heading, each
written to stand alone because retrieval serves them one at a time.

Bias throughout: the things models get *wrong* about WordPress, not the
things they already know. Generic PHP is not in here; WordPress-specific
behaviour, ordering, and security discipline is.

## Actions versus filters — the distinction that breaks sites

An action runs your code at a point in execution and returns nothing:
`add_action('init', 'myplugin_register')`. A filter receives a value,
must return a (possibly modified) value, and breaks the site subtly if
it forgets to — `the_title` returning null empties every title on the
site. Choosing wrongly is the classic defect: acting in a filter, or
forgetting the return.

## Hook priority and argument count

`add_action($hook, $callback, $priority = 10, $accepted_args = 1)`.
Lower priority runs earlier. The fourth argument is the trap: a callback
declared with three parameters but registered without `$accepted_args`
receives only the first, and PHP 8 throws an ArgumentCountError rather
than warning. Always pass it when your callback takes more than one
argument.

## The load-order spine

`muplugins_loaded` → `plugins_loaded` → `setup_theme` →
`after_setup_theme` → `init` → `wp_loaded` → (front)
`template_redirect` → `wp_head` → content → `wp_footer`. Register post
types and taxonomies on `init`; theme supports on `after_setup_theme`;
anything depending on another plugin waits for `plugins_loaded`.
Nothing user-facing should run at file scope.

## Why `init` is the wrong place for current_user checks

`init` fires before the query is parsed, and on some requests before the
user is fully resolved for REST/AJAX contexts. Capability checks belong
in the handler that performs the action, not in a global `init` hook —
checking early and acting later is how privilege checks get bypassed by
a different entry point.

## Escape on output, never on input

The rule is escape-late: `esc_html()` for text nodes, `esc_attr()` for
attribute values, `esc_url()` for URLs, `esc_js()` for inline JS
strings, `wp_kses_post()` for content allowed to carry HTML. Escaping on
input and storing the escaped form corrupts the data and double-escapes
on display; escape at the moment of output, every time, even for values
you believe are safe.

## esc_attr vs esc_html — they are not interchangeable

`esc_html()` encodes `< > & " '` for text nodes. `esc_attr()` is for
values inside quotes in a tag. Using `esc_html()` on an attribute leaves
an injection path through quote breaking in some contexts; using
`esc_attr()` on body text renders visible entities. Match the escaper to
the position.

## Sanitize on input, with the field-appropriate function

Every value from `$_POST`, `$_GET`, `$_REQUEST` or `$_COOKIE` is
untrusted. `wp_unslash()` first (WordPress adds slashes to superglobals),
then `sanitize_text_field()`, `sanitize_email()`, `sanitize_key()`,
`absint()`, `sanitize_textarea_field()`, or `wp_kses_post()` as the field
demands. `intval()` on an id that must be positive is weaker than
`absint()`.

## $wpdb->prepare, always, without exception

```php
$rows = $wpdb->get_results( $wpdb->prepare(
    "SELECT * FROM {$wpdb->prefix}myplugin_items
     WHERE customer_id = %d AND status = %s LIMIT %d",
    $customer_id, $status, $limit
) );
```
Placeholders are `%d`, `%s`, `%f`. The table prefix comes from
`$wpdb->prefix`, never hardcoded `wp_`. Identifiers (table and column
names) cannot be parameterised — validate them against an allowlist
instead of interpolating.

## LIKE queries need esc_like before prepare

`$wpdb->esc_like($term)` escapes `%` and `_` so a user's search string
cannot turn into a wildcard, then the result still goes through
`prepare`: `$wpdb->prepare("... LIKE %s", '%' . $wpdb->esc_like($t) . '%')`.
Skipping `esc_like` is a slow-query and data-leak vector, not just a
correctness bug.

## The nonce round trip, and what a nonce is not

Output `wp_nonce_field('myplugin_save_item', 'myplugin_nonce')` inside
the form. Verify with `wp_verify_nonce(sanitize_key($_POST['myplugin_nonce']),
'myplugin_save_item')` or `check_admin_referer()`. Then still call
`current_user_can()`. A nonce proves the request came from your form and
is recent — it proves *intent*, never *permission*. Nonce without a
capability check is the single most common plugin-review rejection.

## Capability checks use capabilities, not roles

`current_user_can('edit_posts')` — never `current_user_can('editor')`.
Roles are bundles of capabilities that sites customise; checking a role
name breaks on any site that renamed or remapped roles, and silently
grants or denies the wrong people. For per-object checks pass the id:
`current_user_can('edit_post', $post_id)`.

## Nonces expire; that is a feature with a consequence

A nonce is valid for 12–24 hours (two ticks of `nonce_life`). A form
left open overnight fails verification, so a good handler responds with
a re-authentication prompt rather than a fatal error. For long-lived
admin screens, refresh the nonce via `wp_refresh_nonces` heartbeat.

## Enqueue, never echo, scripts and styles

Register on `wp_enqueue_scripts` (front) or `admin_enqueue_scripts`
(admin) with `wp_enqueue_script($handle, $src, $deps, $ver, $in_footer)`.
A `<script>` echoed into `wp_head` defeats dependency ordering,
concatenation, deferral, and any plugin that manages assets. Always pass
a real `$ver` for cache-busting — `filemtime()` on the file is the
honest choice in development.

## Passing data to JavaScript

`wp_localize_script($handle, 'MyPluginData', ['ajaxUrl' => admin_url('admin-ajax.php'), 'nonce' => wp_create_nonce('myplugin')])`
attaches data to an enqueued handle. Modern alternative:
`wp_add_inline_script($handle, 'const X = ' . wp_json_encode($data), 'before')`.
Echoing a `<script>` block with raw PHP interpolation is both an
ordering bug and an XSS vector.

## Custom post types are registered on init, flushed on activation

`register_post_type('myplugin_book', ['public' => true, 'show_in_rest' => true, 'supports' => [...], 'has_archive' => true])`
belongs on `init`. `show_in_rest` is required for the block editor.
Flush rewrite rules **only** in `register_activation_hook` — calling
`flush_rewrite_rules()` on every load is a well-known performance defect
that rewrites the option on each request.

## Post type and taxonomy names have hard limits

Post type keys are max 20 characters, taxonomy keys max 32, both
lowercase with only letters, numbers, underscores and dashes. Prefix
them — an unprefixed `book` collides with any other plugin doing the
same, and the collision is silent until content disappears.

## WP_Query: the arguments that matter for correctness

`new WP_Query(['post_type' => 'book', 'posts_per_page' => 10, 'no_found_rows' => true, 'update_post_meta_cache' => false])`.
`no_found_rows` skips the expensive `SQL_CALC_FOUND_ROWS` when you do
not need pagination. `posts_per_page => -1` is a production incident
waiting for a large site — always bound it. Always `wp_reset_postdata()`
after a secondary loop.

## meta_query is expensive; taxonomies are indexed

Filtering by post meta joins `postmeta`, which has no index on
`meta_value`. For anything users filter or browse by, model it as a
taxonomy instead — taxonomies are indexed and cached. Reach for
`meta_query` for incidental attributes, not for primary navigation.

## The options API and autoload

`update_option($name, $value, $autoload)` — the third argument decides
whether the value loads on **every single request**. Large or rarely-read
options must pass `false`. A plugin storing a big blob as an autoloaded
option is one of the most common causes of a slow site, and it is
invisible until someone profiles `alloptions`.

## Transients for expensive results

`get_transient()` / `set_transient('myplugin_stats', $data, HOUR_IN_SECONDS)`
cache expensive queries and remote calls with automatic expiry. On sites
with a persistent object cache these live in memory; without one they
are autoload-exempt options. Never hand-roll caching into options —
those autoload.

## The Settings API shape

`add_options_page()` on `admin_menu`; `register_setting($group, $name,
['sanitize_callback' => 'myplugin_sanitize'])` plus `add_settings_section`
and `add_settings_field` on `admin_init`; `settings_fields($group)` and
`do_settings_sections($page)` in the form. The sanitize callback is
where every option value gets cleaned — it is the security boundary for
the whole settings screen.

## REST API routes need a real permission_callback

`register_rest_route('myplugin/v1', '/items', ['methods' => 'GET',
'callback' => 'myplugin_get_items', 'permission_callback' => function () {
return current_user_can('edit_posts'); }])` on `rest_api_init`. Since
WordPress 5.5 a missing `permission_callback` triggers a notice, and
`'__return_true'` must be a deliberate, commented decision — it makes
the endpoint public to the entire internet.

## REST arguments are validated in the route, not the callback

Declare `'args' => ['id' => ['required' => true, 'validate_callback' =>
fn($v) => is_numeric($v), 'sanitize_callback' => 'absint']]`. Validation
declared on the route runs before your callback and returns a proper
`WP_Error` with a 400; validating inside the callback duplicates work
the framework already does and usually returns the wrong status.

## Admin-AJAX, and why REST is usually better

Legacy: `add_action('wp_ajax_myplugin_do', ...)` for logged-in users and
`wp_ajax_nopriv_myplugin_do` for anonymous — registering only the first
and then wondering why logged-out users get 0 is a classic. Verify with
`check_ajax_referer('myplugin', 'nonce')`. New code should prefer the
REST API: typed responses, real status codes, and permission callbacks.

## Return WP_Error, not false, from anything a caller must diagnose

`return new WP_Error('myplugin_invalid_sku', __('SKU not found', 'myplugin'), ['status' => 404]);`
A bare `false` tells the caller nothing. In REST callbacks the `status`
in the data array becomes the HTTP status, so a `WP_Error` is the
correct way to produce a 404 or 409.

## Activation, deactivation, uninstall are three different things

`register_activation_hook` creates tables and flushes rewrites.
`register_deactivation_hook` clears scheduled events — never deletes
data, because deactivation is often temporary. Deletion cleanup belongs
in `uninstall.php` (or `register_uninstall_hook`) and removes options,
tables, post meta and transients. Leaving orphaned data behind is a
review-flag; deleting data on *deactivation* is worse.

## dbDelta for schema, with its formatting rules

Table creation uses `dbDelta($sql)` after `require_once ABSPATH . 'wp-admin/includes/upgrade.php'`.
`dbDelta` parses the SQL with a fussy parser: two spaces after `PRIMARY
KEY`, one space between `KEY` and the column list, field types
lowercase, and each field on its own line. Store a schema version in an
option and run `dbDelta` on upgrade, not only on activation.

## Scheduled events run on traffic, not on time

`wp_schedule_event(time(), 'hourly', 'myplugin_sync')` on activation,
`add_action('myplugin_sync', 'myplugin_do_sync')` for the worker,
`wp_clear_scheduled_hook('myplugin_sync')` on deactivation. WP-Cron
fires on page loads, so a low-traffic site runs late and a high-traffic
one can run concurrently — make the worker idempotent, and for
time-critical schedules disable WP-Cron and call `wp-cron.php` from a
real cron.

## Remote requests use the HTTP API, not cURL

`wp_remote_get($url, ['timeout' => 10])`, then `is_wp_error($response)`,
then `wp_remote_retrieve_response_code()` and
`wp_remote_retrieve_body()`. The HTTP API respects site proxy settings,
filters, and blocked-host configuration; raw cURL bypasses all of it and
fails on hosts that restrict outbound traffic. Always set a timeout —
the default can hang a page load.

## Internationalisation, and the text domain trap

Wrap user-facing strings: `__('Save', 'myplugin')`, `esc_html__()`,
`_e()`, `_n()` for plurals, `_x()` for context. The text domain must be
a **literal string**, never a variable or constant — the string-extraction
tools parse source statically and silently skip anything they cannot
read. Since WordPress 4.6 translations load automatically for
plugins.org-hosted plugins.

## Never trust get_the_content in a filter chain

`the_content` runs a long filter chain (wpautop, shortcodes, embeds).
Applying your own filter at default priority can run before or after
formatting depending on registration order. Use an explicit priority and
be aware that returning altered HTML here affects feeds, excerpts and
the REST response too.

## Enqueue conditionally — not everything on every page

`wp_enqueue_script` inside `wp_enqueue_scripts` still loads on every
page unless you gate it: `if (is_singular('book')) { ... }` on the
front, or check `$hook_suffix` in `admin_enqueue_scripts`. Loading an
admin bundle on every admin screen is the most common cause of "this
plugin slows down wp-admin".

## Options, post meta and user meta have different scopes

Site-wide config → options. Per-post data → `update_post_meta()`.
Per-user data → `update_user_meta()`. In multisite, `get_option()` is
per-site and `get_site_option()` is network-wide; a plugin storing
network config in `get_option` breaks the moment it is network-activated.

## Protect meta keys that should not be editable via REST

Meta keys beginning with `_` are protected and hidden from the default
meta UI, but that alone does not stop REST writes. Use
`register_post_meta($post_type, $key, ['show_in_rest' => true,
'single' => true, 'auth_callback' => fn() => current_user_can('edit_posts')])`
so the capability is enforced wherever the write comes from.

## Shortcodes return, never echo

`add_shortcode('recent_books', function ($atts) { $a =
shortcode_atts(['count' => 5], $atts, 'recent_books'); ob_start(); /* … */
return ob_get_clean(); });` A shortcode that echoes prints at the top of
the page instead of in place, because the content filter captures return
values only. `shortcode_atts` supplies defaults and drops unknown keys.

## Blocks: register once, in PHP and JS agreement

`register_block_type(__DIR__ . '/build/my-block')` reads `block.json`,
which is the single source of truth for the block's name, attributes,
editor script and style. Registering attributes in JS that disagree with
`block.json` produces blocks that validate in the editor and break on
the front end.

## Dynamic blocks render in PHP at request time

A block with a `render_callback` (or `render` in `block.json`) is
rendered server-side on every view, so it always reflects current data —
correct for lists of posts. A static block stores its markup in post
content and is fast but frozen. Choosing static for dynamic data is why
"my block does not update" is a support ticket.

## WooCommerce extends through hooks, never template edits

Product data via `woocommerce_product_get_*` filters; checkout fields
via `woocommerce_checkout_fields`; order lifecycle via
`woocommerce_order_status_changed`; cart totals via
`woocommerce_cart_calculate_fees`. Template overrides belong in the
**theme** under `woocommerce/`, never in the plugin directory — plugin
template edits are lost on update.

## WooCommerce CRUD objects, not direct post meta

Since WooCommerce 3.0 read and write through `wc_get_product()`,
`$product->get_price()`, `$product->set_price()`, `$product->save()` and
`wc_get_order()`. Reading `_price` post meta directly bypasses the data
store, breaks under HPOS (High-Performance Order Storage, where orders
live in custom tables rather than posts), and silently returns stale
values.

## The prefix rule applies to everything global

Functions, classes, constants, options, transients, meta keys, script
and style handles, hooks, and shortcodes all share one global namespace
with every other plugin on the site. Prefix all of them (`myplugin_`) or
use a PHP namespace plus prefixed strings. An unprefixed
`add_shortcode('button', …)` will collide.

## Debugging: WP_DEBUG, and the log nobody reads

`define('WP_DEBUG', true); define('WP_DEBUG_LOG', true);
define('WP_DEBUG_DISPLAY', false);` in `wp-config.php` writes to
`wp-content/debug.log` without breaking AJAX and REST responses with
stray output. `WP_DEBUG_DISPLAY` left true is why "my REST endpoint
returns invalid JSON" — a notice printed before the payload.

## Query Monitor is the profiler, not var_dump

For diagnosing slow admin screens, duplicate queries, failed HTTP
requests and hook ordering, Query Monitor shows what actually ran.
`var_dump` in a hook prints into whatever buffer is open and often
corrupts AJAX responses. `error_log(print_r($x, true))` is the safe
alternative when you must print.

## Multisite: network-activated plugins run on every site

Under `is_multisite()`, `switch_to_blog()` / `restore_current_blog()`
are required around cross-site work, and every `switch_to_blog` must be
paired. `$wpdb->prefix` changes per site while `$wpdb->base_prefix`
stays constant — a plugin that caches `$wpdb->prefix` in a static
variable writes to the wrong site's table after a switch.

## Direct file writes go through WP_Filesystem

`WP_Filesystem()` then `$wp_filesystem->put_contents()` respects the
site's configured filesystem method (direct, FTP, SSH) and ownership
rules. `file_put_contents()` works on your machine and fails on managed
hosts that run PHP as a different user than the one owning the files.

## Autoloading and the plugin header

The main plugin file carries the header block (`Plugin Name`, `Version`,
`Requires at least`, `Requires PHP`, `Text Domain`, `License`). WordPress
parses only the first 8 KB of that file for headers, so a large file can
hide its own header. Keep the main file thin: header, guard against
direct access (`defined('ABSPATH') || exit;`), and bootstrap.

## Never dangerouslySet unfiltered HTML from user input

`wp_kses_post()` allows the tag set a post may contain;
`wp_kses($html, $allowed)` lets you specify. Storing raw HTML from a
non-`unfiltered_html` user and echoing it is stored XSS. Only
administrators on single-site installs have `unfiltered_html` — and on
multisite, only super admins.
