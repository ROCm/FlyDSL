function fixBreadcrumbItems() {
    const breadcrumbItems = $("li.breadcrumb-item:not(.breadcrumb-home,.active)");
    const breadcrumbBox = $("ul.bd-breadcrumbs")
    breadcrumbBox.data("maxWidth", 0)
    function adjustLength(item, container, factor, getTextItem) {
        const textItem = getTextItem(item);
        const lines = (x) => Math.round(x.height() / parseFloat(x.css('line-height').replace('px', '')))
        function getMaxWidth(container, itm) {
            const startLines = lines(container);
            const initialText = itm.text();
            let maxWidth = container.width();
            while (lines(container) == startLines) {
                itm.text(itm.text() + "\u200B.");
                if (container.width() > maxWidth) {
                    maxWidth = container.width()
                }
            }
            itm.text(initialText);
            return maxWidth;
        }
        const containerMaxWidth = container.data("maxWidth") || getMaxWidth(container, textItem);
        container.data("maxWidth", containerMaxWidth)
        const fullText = item.data("fullText") || textItem.text();
        item.data("fullText", fullText);
        textItem.text(fullText)
        if (lines(item) == 1 && item.width() < containerMaxWidth * factor) {
            return;
        }
        const words = fullText.split(/\s/);
        let newText = words[0];
        for (let i = 1; i < words.length; i++) {
            textItem.text(newText + " " + words[i] + "...");
            if (lines(item) == 1 && item.width() < containerMaxWidth * factor) {
                newText += " " + words[i];
            } else {
                break;
            }
        }
        newText += "...";
        textItem.text(newText);
    }
    breadcrumbItems.each(function () {
        adjustLength($(this), breadcrumbBox, 0.82 * (breadcrumbItems.length <= 2 ? 1 : 0.5), x => x.children('a'))
    })
    adjustLength($("li.breadcrumb-item.active"), breadcrumbBox, 0.95, x => x);
    breadcrumbBox.data("maxWidth", 0);
}

$(document).ready(function () {
    if (window.ResizeObserver) {
        document.body.addEventListener("bodyresize", event => {
            const { contentRect } = event.detail;
            const { width } = contentRect;
            if (window.prevWidth) {

            }
            if ((window.prevWidth && window.prevWidth > 960) && width < 960) {
                $("input#__primary").prop("checked", false);
            }
            window.prevWidth = width;
            fixBreadcrumbItems();
        })

        const onResizeCallback = (() => {
            let initial = true;
            let timeout;
            return entries => {
                if (initial) {
                    initial = false;
                    return;
                }
                clearTimeout(timeout)
                timeout = setTimeout(() => {
                    for (const entry of entries) {
                        const event = new CustomEvent('bodyresize', {
                            detail: entry
                        });
                        entry.target.dispatchEvent(event);
                    }
                }, 200);
            }
        })()

        window.resizeObserver = new ResizeObserver(onResizeCallback)
        window.resizeObserver.observe(document.body);
    } else {
        console.error("ResizeObserver not supported.")
    }
    fixBreadcrumbItems();
})
