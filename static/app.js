const marketDropdown = document.getElementById("market-select");
const drugDropdown = document.getElementById("drug-select");
const submitBtn = document.getElementById("submit-btn");
const tableSection = document.getElementById("table-section");
const tableHead = document.querySelector("#data-table thead tr")
const tableBody = document.querySelector("#data-table tbody")
const graphSection = document.getElementById("graph-section");

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
    try {
        //getting data from the backend----
        const response = await fetch(`${API_BASE}/filter`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ market_option: market, selected_drugs: selectedDrugs })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        //must return the table of results-----this data will be displayed
        let data = await response.json();
        console.log(data);
        console.log(typeof data);

        data= JSON.parse(data);
        console.log(typeof data);
        
        //checks if the data is array and not null/undefined
        if (Array.isArray(data) && data.length > 0) {
            renderTable(data);
        } else {
            console.warn("No data to display in the table.");
            tableSection.style.display = 'none'; // Hide table if no data
        }


        // Call function to fetch and display the graph
        await fetchAndDisplayGraphs({ market_option: market, selected_drugs: selectedDrugs });

    } catch (error) {
        console.error("Error fetching data:", error);
        alert("Failed to fetch data. Please try again later.");
    }



    //console.log(selectedDrugs);
});


function renderTable(data) {
    tableHead.innerHTML = "";
    tableBody.innerHTML = "";

    if (data.length > 0) {
        tableSection.style.display = "block";//make the table visible

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
    }

}


async function fetchAndDisplayGraphs(filterData) {
    try {
        // Fetch graphs from the backend
        const response = await fetch(`${API_BASE}/graphs`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(filterData)
        });

        if (!response.ok) {
            throw new Error(`Graph API Error: ${response.status} ${response.statusText}`);
        }

        // Handle the response (assuming backend sends an array of image blobs)
        const blobs = await response.json(); // to be adjusted based on the response format
        //expected format is array of images
        graphSection.innerHTML = ""; // Clear previous graphs if any

        if (blobs.length === 0) {
            graphSection.style.display = 'none';
            return;
        }

        graphSection.style.display = 'block';
        // Dynamically create and append images for all graphs

        //checks if blob contains array only and not null/undefined
        if (blobs && Array.isArray(blobs) && blobs.length > 0) {
            blobs.forEach((blobData, index) => {
                // const imageBlob = new Blob([blobData], { type: "image/png" });
                // const imageUrl = URL.createObjectURL(imageBlob);
                // img.src = imageUrl;
                const img = document.createElement("img");                
                img.src = `data:image/png;base64,${blobData}`;
                img.alt = `Graph ${index + 1}`;
                img.style.maxWidth = "100%";
                img.style.marginBottom = "20px";
                graphSection.appendChild(img);
            });
        } else {
            graphSection.style.display = 'none'; // Hide the graph section
            console.warn("No graphs available to display.");
        }

    } catch (error) {
        console.error("Error fetching graphs:", error);
        alert("Failed to fetch graphs. Please try again later.");
    }
}
