document.addEventListener('mouseup', function () {
  var selectedText = window.getSelection().toString();
  if (selectedText !== '') {
    // Highlight the selected text visually (optional)
    // Send the selected text to the background script for analysis
    chrome.runtime.sendMessage({ action: 'analyze-highlighted-text' });
  }
});
