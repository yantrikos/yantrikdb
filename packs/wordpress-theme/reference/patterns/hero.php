<?php
/**
 * Title: Editorial hero
 * Slug: field-notes/hero
 * Categories: featured, banner
 * Viewport Width: 1400
 *
 * Patterns ARE PHP and are executed, which is exactly why dynamic values
 * belong here rather than in a template .html file. This one only needs
 * translation, but the same rule covers home_url(), the current year and
 * anything else a template cannot express.
 */
?>
<!-- wp:group {"align":"full","backgroundColor":"tint","style":{"spacing":{"padding":{"top":"var:preset|spacing|70","bottom":"var:preset|spacing|70"}}},"layout":{"type":"constrained"}} -->
<div class="wp-block-group alignfull has-tint-background-color has-background" style="padding-top:var(--wp--preset--spacing--70);padding-bottom:var(--wp--preset--spacing--70)">
	<!-- wp:paragraph {"className":"eyebrow"} -->
	<p class="eyebrow"><?php echo esc_html__( 'The Journal', 'field-notes' ); ?></p>
	<!-- /wp:paragraph -->
	<!-- wp:heading {"level":1} -->
	<h1 class="wp-block-heading"><?php echo esc_html__( 'Notes from the measurement floor', 'field-notes' ); ?></h1>
	<!-- /wp:heading -->
</div>
<!-- /wp:group -->
