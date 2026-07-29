<?php
/**
 * Field Notes — theme setup.
 *
 * Deliberately almost empty: the palette, type scale, spacing scale and
 * every default style live in theme.json, where the site editor can see
 * them. Duplicating any of that here would create a second source of
 * truth that theme.json silently wins.
 *
 * What cannot live in theme.json is this enqueue. A block theme's
 * `style.css` is NOT loaded automatically — it is a manifest that
 * WordPress reads for the theme header, and its CSS reaches the page only
 * if something enqueues it. The first version of this theme had no
 * functions.php at all, so every optical correction, focus ring, hover
 * state and tap-target rule in style.css was inert. The page still looked
 * designed, because theme.json was carrying it, which is exactly what
 * made the omission hard to notice.
 *
 * @package field-notes
 */

defined( 'ABSPATH' ) || exit;

add_action(
	'wp_enqueue_scripts',
	static function () {
		wp_enqueue_style(
			'field-notes',
			get_stylesheet_uri(),
			array(),
			wp_get_theme()->get( 'Version' )
		);
	}
);

add_action(
	'after_setup_theme',
	static function () {
		// Core block default styles. Everything else a classic theme
		// declared here — colour palette, font sizes, content width,
		// editor styles — is theme.json's job now.
		add_theme_support( 'wp-block-styles' );
		add_theme_support( 'custom-logo', array( 'height' => 48, 'flex-width' => true ) );

		// theme.json reaches the editor on its own; style.css does not.
		add_editor_style( 'style.css' );
	}
);
