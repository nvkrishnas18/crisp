import pandas as pd
import numpy as np
from typing import TypedDict, Any, Dict

import io
import sys

def execute_generated_code(code, df):
    print("Started executing generated code")
    # Clean the code by removing block markers and spaces
    # clean_code = code.strip('```').strip()  
    clean_code = code.replace('```','###')
    clean_code = clean_code.replace("fig.show()","# fig.show()")
    print('111::',clean_code)
    if isinstance(df, pd.DataFrame) and 'Make' in df.columns:
        print("DEBUGGING - Sample Make values:", df['Make'].unique()[:5])
        print("DEBUGGING - Make counts before filtering:")
        print(df['Make'].value_counts().head(10))
    # Redirect stdout to capture print statements
    # output_buffer = io.StringIO()
    # sys.stdout = output_buffer  
    local_scope = {"df": df}

    print('222')

    try:
        print('444')
        exec(clean_code, {}, local_scope)
        print('555')
        if 'final_results' in local_scope:
            print("DEBUGGING - final_results keys:", list(local_scope['final_results'].keys()))
            
    except Exception as e:
        print("Error executing generated code::",e)
        return f"Error during execution: {e}"
    finally:
        print("finally")
        # Reset stdout to default after execution
        # sys.stdout = sys.__stdout__
    print("befre return")
    # Return both the output captured and the local scope (for plots or other variables)
    # output = output_buffer.getvalue()
    print("Ended executing generated code")
    return local_scope

def convert_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, pd.Series):
        return obj.to_json(orient='index')
    elif isinstance(obj, pd.DataFrame):
        return obj.to_json(orient='records')
    elif isinstance(obj, dict):
        return {key: convert_arrays_to_lists(value) for key, value in obj.items()}  
    elif isinstance(obj, list):
        return [convert_arrays_to_lists(item) for item in obj]  
    else:
        return obj  # Return as is if it's not an array/series