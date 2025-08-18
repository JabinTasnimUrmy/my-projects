document.addEventListener('DOMContentLoaded', () => {
    const searchButton = document.getElementById('search-button');
    const ideaInput = document.getElementById('idea-input');
    const resultsContainer = document.getElementById('results-container');
    const loader = document.getElementById('loader');

    searchButton.addEventListener('click', async () => {
        const idea = ideaInput.value.trim();
        if (!idea) {
            alert('Please enter a project idea.');
            return;
        }

        loader.classList.remove('hidden');
        resultsContainer.innerHTML = '';
        searchButton.disabled = true;

        try {
            const response = await fetch('http://127.0.0.1:5000/get_suggestion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ idea: idea }),
            });

            if (!response.ok) { throw new Error('Network response was not ok'); }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            resultsContainer.innerHTML = `<p style="color: red;">An error occurred. Please check the browser console (F12) and the terminal running app.py for more details.</p>`;
        } finally {
            loader.classList.add('hidden');
            searchButton.disabled = false;
        }
    });

    function displayResults(data) {
        if (data.type === 'error') {
            resultsContainer.innerHTML = `<p style="color: orange;">${data.message}</p>`;
            return;
        }

        const project = data.project;
        const cardClass = data.unique ? 'unique-idea' : 'matched-idea';
        const headerText = data.unique 
            ? '💡 Your idea seems unique! Here is the closest match I found:' 
            : '✅ Found a strong match!';

        const resultHTML = `
            <div class="result-card ${cardClass}">
                <p class="result-header">${headerText}</p>
                <p><strong>Project:</strong> <a href="${project.url}" target="_blank">${project.name}</a></p>
                <p><strong>Description:</strong> ${project.description || 'N/A'}</p>
                ${project.stars !== undefined ? `<p><strong>⭐ Stars:</strong> ${project.stars}</p>` : ''}
                ${project.language ? `<p><strong>Language:</strong> ${project.language}</p>` : ''}
            </div>
        `;
        resultsContainer.innerHTML = resultHTML;
    }
});