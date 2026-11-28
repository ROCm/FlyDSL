function renameVersionLinks() {
    $('div.rst-other-versions dl:first-child a').each( function () {
        const text = $(this).text();
        const versionRegEx = /^.*((?:[0-9]+\.){2}[0-9]+).*$/g;
        if (versionRegEx.test(text)) {
            $(this).text(text.replace(versionRegEx, '$1'));
        }
    })
}

function waitForSelector(selector, callback, backoff=100, max=15) {
    let tries = 0;
    const waitInterval = setInterval(() => {
        if ($(selector).length > 0) {
            callback()
            clearInterval(waitInterval)
        } else {
            if (tries++ > max) {
                clearInterval(waitInterval)
            }
        }
    }, backoff)
}

$(document).ready(() => {
    waitForSelector('div.rst-versions', renameVersionLinks);
})
