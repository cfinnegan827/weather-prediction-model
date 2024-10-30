import pandas as pd
def abs_error(x_acc, x_pred):
    print(f"Absolute Error: {abs(x_acc - x_pred)}")
    return abs(x_acc - x_pred)

def perc_error(x_abserr,x_pred):
    print(f"Percent Error: {round(x_abserr/x_pred, ndigits=4) * 100}")
    return round(x_abserr/x_pred, ndigits=2)

def avg_perc_error(x_pred_list):
    cum_val = 0
    for perc in x_pred_list:
        cum_val += perc
    return round(cum_val/len(x_pred_list), 4)
def c_to_f(degree):
    return degree * (9/5) + 32

def get_data(date):
    df = pd.read_csv('temp.csv')
    specific_value = date
    result = df[df.iloc[:, 0] == specific_value]
    print(result)
    return result

