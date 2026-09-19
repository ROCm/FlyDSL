// Upgrade banner links marked with `data-rocm-banner-latest-link` so they point
// at the same page on the latest version instead of a fixed docs root.
//
// The built-in "old version" / release-candidate / development banners (see
// theme._update_banner) ship a link with a fixed `.../en/latest/` href as a
// no-JS fallback. ROCm docs are served by Read the Docs under
// `.../en/<version>/<page>`, so when JS runs we rewrite the href to the same
// <page> under `latest`, preserving the project prefix.
//
// On click, we first check whether that page actually exists on `latest`
// (pages get renamed or removed between versions, and a redirect may be
// missing). If the page is gone, we send the reader to the project's landing
// page on `latest` instead of Read the Docs' own 404. Existing redirects still
// work: the HEAD request follows them, so a redirected page counts as present.
//
// Only links carrying the `data-rocm-banner-latest-link` attribute are touched,
// so a project that overrides `announcement` with its own link is left alone
// unless it opts in by adding the attribute.

(function () {
    function latestTargets(loc) {
        const match = loc.pathname.match(/^(.*\/en\/)[^/]+(\/.*)?$/);
        if (!match) {
            return null;
        }
        const base = loc.origin + match[1] + "latest";
        const pagePath = match[2] || "/";
        return {
            // Matching page on the latest version (without the fragment).
            page: base + pagePath,
            // Preserve the current fragment when navigating to the page.
            hash: loc.hash,
            // Project landing page on the latest version, used as the
            // fallback when the matching page no longer exists.
            home: base + "/",
        };
    }

    function navigateToLatest(targets) {
        // Without fetch we cannot check existence; keep the current behavior of
        // navigating to the matching page and letting the browser resolve it.
        if (typeof fetch !== "function") {
            window.location.assign(targets.page + targets.hash);
            return;
        }
        // The HEAD request follows redirects, so a redirected page returns a
        // 2xx and counts as present. A genuine 404 means the page is gone with
        // no redirect: fall back to the project landing page on latest.
        //
        // We match 404 specifically rather than `!response.ok` on purpose. Only
        // a 404 is a definite "page is missing" signal. A transient 5xx (or a
        // 403) does not mean the page is gone, so diverting to the project home
        // would strip the reader's page context for what is likely a temporary
        // problem the actual navigation may not even hit. That mirrors the
        // `catch` below, which treats an inconclusive check by navigating to the
        // page, not home: we only send the reader home on a definite 404.
        fetch(targets.page, { method: "HEAD" })
            .then((response) => {
                if (response.status === 404) {
                    window.location.assign(targets.home);
                } else {
                    window.location.assign(targets.page + targets.hash);
                }
            })
            .catch(() => {
                // Network error: don't trap the reader, navigate to the page.
                window.location.assign(targets.page + targets.hash);
            });
    }

    function rewriteBannerLatestLinks() {
        const targets = latestTargets(window.location);
        if (!targets) {
            return;
        }
        const links = document.querySelectorAll(
            "[data-rocm-banner-latest-link]"
        );
        links.forEach((link) => {
            // Set the href so the no-JS fallback, "open in new tab", and "copy
            // link address" all point at the matching page on latest.
            link.href = targets.page + targets.hash;
            link.addEventListener("click", (event) => {
                // Let the browser handle modified or non-primary clicks (new
                // tab, new window, download) natively via the href.
                if (
                    event.defaultPrevented ||
                    event.button !== 0 ||
                    event.metaKey ||
                    event.ctrlKey ||
                    event.shiftKey ||
                    event.altKey
                ) {
                    return;
                }
                event.preventDefault();
                navigateToLatest(targets);
            });
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded", rewriteBannerLatestLinks
        );
    } else {
        rewriteBannerLatestLinks();
    }
})();
