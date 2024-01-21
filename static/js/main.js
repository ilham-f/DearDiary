document.addEventListener('DOMContentLoaded', function () {
    const textarea = document.getElementById('textarea');
    const suggestionsList = document.getElementById('suggestions');

    textarea.addEventListener('input', function () {
        handleInput();
    });

    textarea.addEventListener('keydown', function (event) {
        // Handle spacebar press separately
        if (event.key === ' ' || event.code === 'Space') {
            handleInput();
        }
    });

    function handleInput() {
        const inputText = textarea.value.toLowerCase();
        const cursorPosition = textarea.selectionStart;

        const prefix = inputText.substring(0, cursorPosition).split(' ').pop();
        const suggestions = getMatchingWords(prefix);

        // Clear existing suggestions
        suggestionsList.innerHTML = '';

        // Populate the suggestions list
        suggestions.forEach(function (word) {
            const li = document.createElement('li');
            li.textContent = word;
            li.addEventListener('click', function () {
                replaceLastWord(word);
            });
            suggestionsList.appendChild(li);
        });
    }

    function getMatchingWords(prefix) {
        // In a real-world scenario, you might fetch this list from a server.
        const wordList = ['apple', 'banana', 'cherry', 'date', 'grape', 'kiwi', 'orange', 'pear', 'watermelon'];
        return wordList.filter(word => word.startsWith(prefix));
    }

    function replaceLastWord(replacement) {
        const currentText = textarea.value;
        const cursorPosition = textarea.selectionStart;
        const words = currentText.split(' ');

        let newText = '';
        for (let i = 0; i < words.length; i++) {
            if (i === words.length - 1) {
                newText += replacement;
            } else {
                newText += words[i];
            }
            if (i < words.length - 1) {
                newText += ' ';
            }
        }

        textarea.value = newText;
        textarea.focus();
        suggestionsList.innerHTML = ''; // Clear suggestions after replacing the word
    }
});
