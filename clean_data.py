import numpy as np
import pandas as pd

# ================================================================================================================================================
# ================================================================================================================================================
i = -1
def getincrement():
    global i
    i = i + 1
    return f"f_{i}"

def change_col_names(df, target_col_name):
    cols = list(df.columns)
    cols = [getincrement() if (i != target_col_name) else 'target' for i in cols]
    df.columns = cols
    return df
# ================================================================================================================================================
# ================================================================================================================================================


# ================================================================================================================================================
# ================================================================================================================================================
def create_new_feats(df):
    
    df["Date"] = df.InvoiceDate.apply(lambda x: x.split(" ")[0])
    df["Time"] = df.InvoiceDate.apply(lambda x: x.split(" ")[1])

    df['Year'] = df.Date.apply(lambda x: x.split("-")[0])
    df['Month'] = df.Date.apply(lambda x: x.split("-")[1])
    df['Day'] = df.Date.apply(lambda x: x.split("-")[2])

    df['Hour'] = df.Time.apply(lambda x: x.split(":")[0])
    df['Min'] = df.Time.apply(lambda x: x.split(":")[1])

    df.drop(
        ['Date', 'Time', 'InvoiceDate'],
        axis = 1, inplace=True
    )

    return df
# ================================================================================================================================================
# ================================================================================================================================================

def to_int(df , xcept):
    for colname in df.columns:
        if colname != xcept:
            #print(f'converting {colname}')
            df[colname] = df[colname].apply(int)
    return df

if __name__ == "__main__":

    # read
    df = pd.read_csv('inputs/Train.csv')
    
    """
    + Same invoice number for same cutomer on same day
    + will it add correlation?
    """
    # invoice_nos = df.InvoiceNo.values
    # vals, cnts = np.unique(invoice_nos, return_counts=True)
    # print(f'total unique: {len(vals)}')
    # print(f'total: {len(df)}')
    #
    # print(vals[0], cnts[0])
    # print(df[df.InvoiceNo == vals[0]])
    #
    #
    # print(vals[2], cnts[2])
    # print(df[df.InvoiceNo == vals[2]])


    """
    create new cols for year, month, day, hr, mins
    """
    df = create_new_feats(df)
    
    # swap colnames and values
    df.loc[:, 'UnitPrice'], df.loc[:, 'Min'] = df.loc[:, 'Min'], df.loc[:, 'UnitPrice'] 
    cols = list(df.columns)
    cols[4], cols[11] =  cols[11], cols[4]
    df.columns = cols


    """
    change col names
    """
    df= change_col_names(df, 'UnitPrice')    
    

    """
    typecast to int
    """
    df = to_int(df, xcept='target')


    """
    save to csv
    """
    df.to_csv("inputs/train_clean.csv", index=False)