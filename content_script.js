// Capture highlighted text
document.addEventListener('mouseup', function(event) {
  var selectedText = window.getSelection().toString();
  chrome.runtime.sendMessage({type: "highlightedText", text: selectedText});
});
