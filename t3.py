import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



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
        filtered_df['NPI'] = filtered_df['NPI'].astype(str)
    
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
    df['Total Sum'] = df.filter(regex='|'.join(keywords)).sum(axis=1)
    
    # Filter out rows where the total sum is 0
    df = df[df['Total Sum'] > 0]

    df['Zip code'] = df['Zip code'].astype(str).apply(lambda x: x.zfill(9) if len(x) in [6, 7, 8] else x.zfill(5) if len(x) == 5 else x)

    
    # Sort the DataFrame by 'Total Sum' in descending order
    df = df.sort_values(by='Total Sum', ascending=False)
    
    # Drop the 'Total Sum' column before displaying
    df = df.drop(columns=['Total Sum'])
    
    return df

# Function to generate visualizations based on summed values for selected drugs
def generate_visualizations(df, selected_drugs):
    patients_totals = pd.Series(dtype=float)
    claims_totals = pd.Series(dtype=float)

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
            st.pyplot(fig_pie)
            plt.close(fig_pie)
            
            fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
            ax_bar.bar(totals.index, totals.values, color='skyblue')
            ax_bar.set_xlabel('Payment Type')
            ax_bar.set_ylabel('Total Amount')
            ax_bar.set_title(f"Total Payments for {drug}")
            ax_bar.tick_params(axis='x', rotation=45)
            # st.pyplot(fig_bar)
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
        if data.empty or data.sum() == 0:
            # st.warning(f"No data available for {title}")
            return

        fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
        data.plot(kind='bar', ax=ax_bar, color='skyblue')
        ax_bar.set_title(title)
        ax_bar.set_ylabel(ylabel)
        ax_bar.set_xlabel("Drugs")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    if not patients_totals.empty:
        plot_bar_chart(patients_totals, "Total Patients by Drug", "Number of Patients")

    if not claims_totals.empty:
        plot_bar_chart(claims_totals, "Total Claims by Drug", "Number of Claims(in Millions)")

# Main function to run the Streamlit app
def main():
    logo_path = r"yaali_animal.png"
    st.sidebar.image(logo_path, use_column_width=True)

    # logo_lhs = Image.open("roche.jpg")
    # # logo_middle = Image.open("yaali_animal.png")
    # logo_rhs = Image.open("yaali.png")

    # logo_middle = logo_lhs.resize((300, 50))  # Width: 300px, Height: 150px

    st.image("final.png", use_column_width=True)
    # col1, col2, col3 = st.columns([3, 1, 4])
    # with col1:
    #     st.image(logo_lhs, use_column_width=True)
    # # with col2:
    # #     st.image(logo_middle, use_column_width=True)
    # with col3:
    #     st.image(logo_rhs, use_column_width=True)
    st.link_button("About Yaali", "https://drive.google.com/file/d/15T68XeTxSJ68qm0D75wSEj09H7Rdt_tQ/view")
    st.title("KOL Identification Tool")

    market_option = st.selectbox("Market of Interest", ["GASTRO", "LUPUS"])
    df = load_data(market_option)

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
    selected_drugs = st.multiselect(
        "Drugs of Interest", [all_drugs_option] + drugs[market_option], default=[]
    )

    if all_drugs_option in selected_drugs:
        selected_drugs = drugs[market_option]

    if selected_drugs:
        filtered_df = filter_columns(df, selected_drugs, common_columns)
        filtered_df = sum_and_sort_columns(filtered_df)
        filtered_df = filtered_df.reset_index(drop=True)

        # highlighted_df = filtered_df.style.applymap(lambda x: 'background-color: yellow')
        # st.write(highlighted_df)
        st.write(filtered_df)
        # st.write(filtered_df.to_html(index=False))
        # pd.set_option("styler.render.max_elements", 814476)


        generate_visualizations(filtered_df, selected_drugs)

if __name__ == "__main__":
    main()
