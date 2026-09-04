// Trigger the Read the Docs Addons Search modal when clicking on "Search docs" input from the topnav.
document.querySelector(".search-button-field.search-button__button").addEventListener("focusin", () => {
    const event = new CustomEvent("readthedocs-search-show");
    document.dispatchEvent(event);
 });
