document.addEventListener('DOMContentLoaded', function () {
  var analyzeButton = document.getElementById('analyzeButton');
  analyzeButton.addEventListener('click', function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'analyze-highlighted-text' });
    });
  });
});
