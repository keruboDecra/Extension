document.addEventListener('DOMContentLoaded', function() {
  var highlightBtn = document.getElementById('highlightBtn');
  var analyzeBtn = document.getElementById('analyzeBtn');
  var resultDiv = document.getElementById('result');

  // Highlight text button
  highlightBtn.addEventListener('click', function() {
    // Your highlighting logic goes here
  });

  // Analyze text button
  analyzeBtn.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {type: "analyzeText"}, function(response) {
        resultDiv.textContent = response.result;
      });
    });
  });
});
