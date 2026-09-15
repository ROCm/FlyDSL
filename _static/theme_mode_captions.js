function modifyThemeModeCaptions() {
    var themeSwitchButtons = document.getElementsByClassName("theme-switch-button");
    for (var i = 0; i < themeSwitchButtons.length; i++) {
        themeSwitchButtons[i].setAttribute("data-bs-original-title", document.documentElement.dataset.mode);
    }
}

function addModeListener() {
    const btn = document.getElementsByClassName("theme-switch-button")[0];
    btn.addEventListener("click", modifyThemeModeCaptions);
}

$(addModeListener);
$(window).ajaxComplete(function() {
    setTimeout(() => {
        modifyThemeModeCaptions();
    }, 3000);
})
