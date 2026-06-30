function resetForm() {

    // Clear Title
    document.querySelector('input[name="title"]').value = "";

    // Clear Blog Content
    document.querySelector('textarea[name="content"]').value = "";

    // Hide Result Section
    const result = document.querySelector(".result");
    if(result){
        result.style.display = "none";
    }

    // Hide Loading Animation
    const loading = document.querySelector(".loading");
    if(loading){
        loading.style.display = "none";
    }

}