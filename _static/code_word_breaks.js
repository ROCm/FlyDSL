$(document).ready(() => {
    const copy = async(event) => {
        return await navigator.clipboard.writeText($(event.target).attr('copydata'));
    }

    $('.table td code').each( function () {
        var text = $(this).text()
        $(this).addClass('hovertext')
        $(this).attr('copydata', text)
        $(this).attr('data-hover', "Click to copy.")
        var new_text = text.replaceAll(/_([^\u200B])/g, '_\u200B$1').replaceAll(/([a-z])([A-Z])/g, '$1\u200B$2')
        $(this).text(new_text)
        $(this).click((event) => {
            copy(event)
            $(event.target).attr('data-hover', "Copied!")
            $(event.target).on("mouseleave", () => {
                $(event.target).attr('data-hover', "Click to copy.")
                $(event.target).off("mouseleave")
            })
        })
    })
})
