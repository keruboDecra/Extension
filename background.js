// Receive highlighted text from content script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.type === "highlightedText") {
    // Analyze the text or perform any other operations
    var analyzedText = analyzeText(request.text);
    // Send the analyzed text back to the content script
    sendResponse({result: analyzedText});
  }
});

// Example function to analyze text
function analyzeText(text) {
  // Your analysis logic goes here
  return "Analysis result: " + text.toUpperCase();
}
