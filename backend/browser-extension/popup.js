const ANNOTATE_ENDPOINT = "http://localhost:8501/?url=";

document.getElementById("Annotate").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        console.log(tabs);
        const url = tabs[0].url; // url can be https://arxiv.org/abs/2305.18290
        const pdf_url = url.replace("/abs/", "/pdf/");
        const annotationUrl =  ANNOTATE_ENDPOINT + pdf_url;
        chrome.tabs.create({ url: annotationUrl });
    });
});