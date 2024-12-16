const marketDropdown = document.getElementById("market-select");
const drugDropdown = document.getElementById("drug-select");
const submitBtn = document.getElementById("submit-btn");
const tableSection = document.getElementById("table-section");
const tableHead = document.querySelector("#data-table thead tr")
const tableBody = document.querySelector("#data-table tbody")
const graphSection = document.getElementById("graph-section");
const exportBtn = document.getElementById("export-btn");

const API_BASE = "http://127.0.0.1:8000";//Backend API
//data for the second drop down menu
const drugs = {
    GASTRO: [
        "All", "SKYRIZI", "ENTYVIO", "STELARA", "INFLECTRA",
        "HUMIRA", "ZEPOSIA", "SIMPONI", "RENFLEXIS", "REMICADE", "CIMZIA"
    ],
    LUPUS: [
        "All","SAPHNELO", "LUPKYNIS", "BENLYSTA"
    ]
}

marketDropdown.addEventListener("change", function () {
    const market = marketDropdown.value;
    //populating drugDropdown
    drugDropdown.innerHTML = '<option value="" disabled selected>Select drugs</option>';
    drugDropdown.disabled = true;

    if (drugs[market]) {
        drugs[market].forEach(element => {
            const option = document.createElement("option");
            option.value = element
            option.textContent = element
            drugDropdown.appendChild(option);

        });
        drugDropdown.disabled = false;
    }
    else {
        drugDropdown.disabled = true;
    }

});
submitBtn.addEventListener("click", async () => {
    const market = marketDropdown.value;
    const selectedDrugs = Array.from(drugDropdown.selectedOptions).map(opt => opt.value);
    if (selectedDrugs.length === 0) {
        alert("Please select at least one drug.");
        return; // Prevent further execution
    }

    // Show the spinner before fetching data
    const loadingSpinner = document.getElementById("loading-spinner");
    loadingSpinner.style.display = "block";

    try {
        // Getting data from the backend
        const response = await fetch(`${API_BASE}/filter`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ market_option: market, selected_drugs: selectedDrugs })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        // Parse the response data
        let data = await response.json();
        data = JSON.parse(data); // If it's a JSON string

        if (Array.isArray(data) && data.length > 0) {
            renderTable(data); // Render the table
        } else {
            console.warn("No data to display in the table.");
            tableSection.style.display = 'none'; // Hide the table if no data
            exportBtn.style.display= "none";
        }

        // Fetch and display graphs
        await fetchAndDisplayGraphs({ market_option: market, selected_drugs: selectedDrugs });

    } catch (error) {
        console.error("Error fetching data:", error);
        alert("Failed to fetch data. Please try again later.");
    } finally {
        // Hide the spinner after fetching is complete
        loadingSpinner.style.display = "none";
    }
});
function renderTable(data) {
    tableHead.innerHTML = "";
    tableBody.innerHTML = "";

    if (data.length > 0) {
        //console.log(exportBtn);
        exportBtn.style.display="block";
        tableSection.style.display = "block";//make the table visible
        //console.log(exportBtn.style.display);
        

        //Generate table headers dynamically
        const headers = Object.keys(data[0]);// Get column names from the first row
        headers.forEach(header => {
            const th = document.createElement("th");
            th.textContent = header;
            tableHead.appendChild(th);

        });

        // Populate table rows with data
        data.forEach(row => {
            const tr = document.createElement("tr");
            Object.values(row).forEach(value => {
                const td = document.createElement("td");
                td.textContent = value;
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });

    }
    else {
        tableSection.style.display = 'none';
        exportBtn.style.display="none";
    }

}


async function fetchAndDisplayGraphs(filterData) {
    const loadingSpinner = document.getElementById("loading-spinner");
    loadingSpinner.style.display = "block"; // Show spinner for graphs

    try {
        const response = await fetch(`${API_BASE}/graphs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(filterData)
        });

        if (!response.ok) {
            throw new Error(`Graph API Error: ${response.status} ${response.statusText}`);
        }

        const blobs = await response.json(); // Expected array of base64 strings
        graphSection.innerHTML = ""; // Clear previous graphs

        if (blobs && Array.isArray(blobs) && blobs.length > 0) {
            blobs.forEach((blobData, index) => {
                const img = document.createElement("img");
                img.src = `data:image/png;base64,${blobData}`;
                img.alt = `Graph ${index + 1}`;
                img.style.maxWidth = "100%";
                img.style.marginBottom = "20px";
                graphSection.appendChild(img);
            });
            graphSection.style.display = 'block'; // Show the graph section
        } else {
            graphSection.style.display = 'none'; // Hide if no graphs
        }

    } catch (error) {
        console.error("Error fetching graphs:", error);
        alert("Failed to fetch graphs. Please try again later.");
    } finally {
        // Hide the spinner after graph fetching is complete
        loadingSpinner.style.display = "none";
    }
}

// Function to export table data to CSV

exportBtn.addEventListener('click',()=> {
    filename = 'table_data.csv';
    console.log("button click hua export wala");
    const rows = [];
    // Get table headers
    const headers = Array.from(tableHead.querySelectorAll('th')).map(th => th.textContent);
    rows.push(headers);

    // Get table body rows
    tableBody.querySelectorAll('tr').forEach(tr => {
        const row = Array.from(tr.querySelectorAll('td')).map(td => td.textContent);
        rows.push(row);
    });

    // Convert to CSV
    const csvContent = rows.map(row => row.join(',')).join('\n');

    // Create a Blob and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
});