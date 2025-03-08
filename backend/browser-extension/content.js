function uncheckFollowCompanyCheckbox() {
    const followCompanyCheckbox = document.getElementById("follow-company-checkbox");
    if (followCompanyCheckbox) {
        followCompanyCheckbox.checked = false;
    }
}

window.onload = () => {
    uncheckFollowCompanyCheckbox();
};