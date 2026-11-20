const RTD_SEARCH_EVENT = "readthedocs-search-show";
const SEARCH_FIELD_QUERY = ".search-button-field.search-button__button";

function showRtdSearch() {
    document.dispatchEvent(new CustomEvent(RTD_SEARCH_EVENT));
}

// Trigger the Read the Docs Addons Search modal when focusing on the search
// field in the left sidebar.
const searchField = document.querySelector(SEARCH_FIELD_QUERY);
if (searchField) {
    searchField.addEventListener("focusin", showRtdSearch);
}

// To preserve old search shortcut, open the Read the Docs Addons Search modal
// on Ctrl+K / Cmd+K. Read the Docs Addons Search uses slash by default and
// doesn't need to be handled here.
document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        showRtdSearch();
    }
});
