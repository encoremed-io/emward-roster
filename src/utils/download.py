from io import BytesIO
import pandas as pd
import streamlit as st

def download_excel(df, filename):
    """ Generates an Excel file from a DataFrame and downloads it. """
    buffer = BytesIO()
    df.to_excel(buffer,
                index=not df.index.equals(pd.RangeIndex(len(df))),
                engine="xlsxwriter")
    buffer.seek(0)
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )