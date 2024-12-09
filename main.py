from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from PIL import Image


app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')
graph = []

class FilterRequest(BaseModel):
    market_option : str
    selected_drugs: list

# Function to load data based on the selected market option
def load_data(market_option):
    if market_option == "GASTRO":
        return pd.read_csv("Final_GASTRO.csv")
    elif market_option == "LUPUS":
        return pd.read_csv("Final_LUPUS.csv")
    
# Function to filter columns based on selected drugs
def filter_columns(df, selected_drugs, common_columns):
    drug_columns = []
    for drug in selected_drugs:
        drug_columns.extend([
            f"{drug}_CONSULTING", f"{drug}_EDUCATION", f"{drug}_FOOD&BEVERAGE",
            f"{drug}_GENERAL", f"{drug}_SPEAKER", f"{drug}_TRAVEL",
            f"{drug}_Claims", f"{drug}_Patients",f"{drug}_OTHERS",f"{drug}_OTHERS_GENERAL"
        ])
    available_columns = [col for col in common_columns + drug_columns if col in df.columns]
    
    # Filter the DataFrame to include only the selected columns
    filtered_df = df[available_columns]
    
    # Convert the NPI column to string type
    if 'NPI' in filtered_df.columns:
        filtered_df.loc[:, 'NPI'] = filtered_df['NPI'].astype(str)
    
    # Process the 'Provider Business Mailing Address Postal Code' column
    if 'Provider Business Mailing Address Postal Code' in filtered_df.columns:
        filtered_df['Provider Business Mailing Address Postal Code'] = (
            filtered_df['Provider Business Mailing Address Postal Code']
            .astype(str)
            .str.zfill(5)
            .apply(lambda x: x.zfill(9) if len(x) > 5 else x)
            .apply(lambda x: f"{x[:5]}-{x[5:]}" if len(x) == 9 else x)
        )
    
    return filtered_df

# Function to sum specific columns and sort the DataFrame
def sum_and_sort_columns(df):
    # Keywords to filter columns
    keywords = ['CONSULTING', 'EDUCATION', 'FOOD&BEVERAGE', 'GENERAL', 'SPEAKER', 'TRAVEL', "OTHERS", "OTHERS_GENERAL",'Claims','Patients']
    
    # Calculate the total sum of columns containing the keywords
    df.loc[:, 'Total Sum'] = df.filter(regex='|'.join(keywords)).sum(axis=1)
    
    # Filter out rows where the total sum is 0
    df = df[df['Total Sum'] > 0]

    df.loc[:, 'Zip code'] = df['Zip code'].astype(str).apply(lambda x: x.zfill(9) if len(x) in [6, 7, 8] else x.zfill(5) if len(x) == 5 else x)

    
    # Sort the DataFrame by 'Total Sum' in descending order
    df = df.sort_values(by='Total Sum', ascending=False)
    
    # Drop the 'Total Sum' column before displaying
    df = df.drop(columns=['Total Sum'])
    
    return df

# Function to generate visualizations based on summed values for selected drugs
def generate_visualizations(df, selected_drugs):
    patients_totals = pd.Series(dtype=float)
    claims_totals = pd.Series(dtype=float)
    inner_graph = []

    for drug in selected_drugs:
        columns = {
            'CONSULTING': f"{drug}_CONSULTING",
            'EDUCATION': f"{drug}_EDUCATION",
            'FOOD&BEVERAGE': f"{drug}_FOOD&BEVERAGE",
            'GENERAL': f"{drug}_GENERAL",
            'SPEAKER': f"{drug}_SPEAKER",
            'TRAVEL': f"{drug}_TRAVEL",
            "OTHERS":f"{drug}_OTHERS",
            "OTHERS_GENERAL":f"{drug}_OTHERS_GENERAL"

        }

        available_columns = {label: col for label, col in columns.items() if col in df.columns}
        
        if available_columns:
            totals = df[list(available_columns.values())].sum()
            
            fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
            wedges, texts = ax_pie.pie(
                totals, labels=None, autopct=None, startangle=140,
                pctdistance=0.85, shadow=True, wedgeprops=dict(width=0.3)
            )
            ax_pie.set_title(f"Distribution of Payments for {drug}")
            ax_pie.legend(wedges, totals.index, title="Payment Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig_pie)
            buf.seek(0)
            # img = Image.open(buf)
            inner_graph.append(buf.getvalue())
            buf.close()

            fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
            ax_bar.bar(totals.index, totals.values, color='skyblue')
            ax_bar.set_xlabel('Payment Type')
            ax_bar.set_ylabel('Total Amount')
            ax_bar.set_title(f"Total Payments for {drug}")
            ax_bar.tick_params(axis='x', rotation=45)            
            plt.close(fig_bar)
            

    available_columns = {
        'CLAIMS': f"{drug}_Claims",
        'PATIENTS': f"{drug}_Patients"
    }
    for drug in selected_drugs:
        if available_columns:
            if f"{drug}_Patients" in df.columns:
                patients_totals[drug] = df[f"{drug}_Patients"].fillna(0).sum()
            if f"{drug}_Claims" in df.columns:
                claims_totals[drug] = df[f"{drug}_Claims"].fillna(0).sum()

    def plot_bar_chart(data, title, ylabel):
        global graph
        if data.empty or data.sum() == 0:
            return

        fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
        data.plot(kind='bar', ax=ax_bar, color='skyblue')
        ax_bar.set_title(title)
        ax_bar.set_ylabel(ylabel)
        ax_bar.set_xlabel("Drugs")
        plt.xticks(rotation=45, ha='right')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig_bar)
        buf.seek(0)
        # img = Image.open(buf)
        inner_graph.append(buf.getvalue())
        buf.close()

    if not patients_totals.empty:
        plot_bar_chart(patients_totals, "Total Patients by Drug", "Number of Patients")

    if not claims_totals.empty:
        plot_bar_chart(claims_totals, "Total Claims by Drug", "Number of Claims(in Millions)")

    return inner_graph



    
# Render the home page
@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

# Return the table
@app.post('/filter', response_class=JSONResponse)
async def filter(request: FilterRequest):
    df = load_data(request.market_option)

    drugs = {
        "GASTRO": [
            "SKYRIZI", "ENTYVIO", "STELARA", "INFLECTRA", 
            "HUMIRA", "ZEPOSIA", "SIMPONI", "RENFLEXIS", "REMICADE", "CIMZIA"
        ],
        "LUPUS": [
            "SAPHNELO", "LUPKYNIS", "BENLYSTA"
        ]
    }

    common_columns = [
        'NPI', 'Last Name', 'First Name',
        'Middle Name', 'Address',
        'Provider Second Line Business Mailing Address',
        'City',
        'State',
        'Zip code',
        'Primary_Classification', 'Primary_Specialization', 'Definition',
        'Notes', 'Display Name', 'Section', 'Secondary_Classification',
        'Secondary_Specialization', 'Definition.1', 'Notes.1', 'Display Name.1',
        'Section.1'
    ]

    all_drugs_option = "All"
    selected_drugs = request.selected_drugs 
    if all_drugs_option in selected_drugs:
        selected_drugs = drugs[request.market_option]

    if selected_drugs:
        filtered_df = filter_columns(df, selected_drugs, common_columns)
        filtered_df = sum_and_sort_columns(filtered_df)
        filtered_df = filtered_df.reset_index(drop=True)

    global graph 
    graph = generate_visualizations(filtered_df, selected_drugs)
    resp = filtered_df.to_json(orient='records')
    return resp

# Return the Graphs
@app.post('/graphs')
async def graphs(request: FilterRequest):
    return [base64.b64encode(img).decode("utf-8") for img in graph]
    