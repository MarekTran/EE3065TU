# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

# %%
# Capacity and Unavailability 
df = pd.read_csv('./data/Cgens_Ugens.csv')
# Capacity of generators
Cgens = df['Cgens'].values.tolist()
# Probability of failure of generators (unavailability)
Ugens = df['Ugens'].values.tolist()
# Wind DF
df = pd.read_csv('./data/load_Pwind.csv')
# Hourly load
NLh_load = df['NLh_load'].values.tolist()
# Power of first offshore wind scenario
P_off_wind1 = df['P_off_wind1'].values.tolist()
# Power of second offshore wind scenario
P_off_wind2 = df['P_off_wind2'].values.tolist()
# Power of third offshore wind scenario
P_off_wind3 = df['P_off_wind3'].values.tolist()
# Round to a multiple of this (50)
P_round = 50

# %%
# PLOT LOAD CURVE
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(NLh_load)), y=NLh_load, name='Load'))
fig.update_layout(title='Load', xaxis_title='Index', yaxis_title='Load (MW)')
# Size 800x800
fig.update_layout(width=800,height=800)
fig.show()


# %% [markdown]
# ## Question 4
# + What is the size of the smallest and largest generator?
# + What are the smallest and largest unavailabilities?
# + What is the total installed capacity of the generation system?
# + What is the peak load?

# %%
# Smallest generator
smallest_generator = np.min(Cgens)
print('Smallest generator: ', smallest_generator)
# Largest generator
largest_generator = np.max(Cgens)
print('Largest generator: ', largest_generator)
# Total capacity
Ctot = np.sum(Cgens)
print('Total capacity: ', Ctot)
# Smallest unavailability
smallest_unavailability = np.min(Ugens)
print('Smallest unavailability: ', smallest_unavailability)
# Largest unavailability
largest_unavailability = np.max(Ugens)
print('Largest unavailability: ', largest_unavailability)

# %%
def COPT_function(Cgens,Ugens,P_round):
    # print(Cgens)
    # print(Ugens)
    # assert Ugens has no negative values
    assert np.all(np.array(Ugens) >= 0)
    # Eduardo Jerez, based on Bart Tuinema's Matlab code
    # COPT calculation
    # COPT = COPT_function(Cgens,Ugens,P_round)
    # Cgens = unit capacities
    # Ugens = Generator unavailabilities (FOR's)
    # P_round = capacities are rounded to this value

    Cgens = np.multiply(P_round,np.round(np.divide(Cgens,P_round)))  # round capacities
    P_total = sum(Cgens)  # total capacity
    n_units = len(Cgens)  # number of units

    COPT = np.zeros([np.int_(P_total/P_round), 2])
    COPT[:,0] = range(0,np.int_(P_total),P_round)
    COPT[0,1] = 1
    for i in range(n_units):
        COPT2 = np.multiply(COPT[:,1],Ugens[i])  # COPT when unit i is off
        COPT[:,1] = np.multiply(COPT[:,1],(1-Ugens[i]))  # COPT when unit i is on
        COPT[:,1] = np.add(COPT[:,1], np.hstack([np.zeros([np.int_(Cgens[i]/P_round)]), COPT2[0:-np.int_((Cgens[i]/P_round))]]))    #COPT2 is shifted down by the capacity of the generator

    #COPT = [COPT zeros(size(COPT[0]),1)]    # adding 3rd column
    #COPT(1,3) = 1
    
    COPT = np.transpose(np.vstack([np.transpose(COPT), np.subtract(1,np.hstack([[0],np.cumsum(COPT[0:-1,1])]))]))
    # Convert to DataFrame and round probabilities to 4 decimals
    COPT = pd.DataFrame(COPT, columns=['Capacity', 'Probability', 'Cumulative Probability'])
    # COPT['Probability'] = COPT['Probability'].round(4)
    # COPT['Cumulative Probability'] = COPT['Cumulative Probability'].round(4)
    return COPT

# COPT_function_old(Cgens,Ugens,P_round)

# %% [markdown]
# ## Question 5
# + LOLP?
# + LOLE?
# + LOLEE 

# %%
def get_LOEE(load, df_COPT):
    # Convert relevant DataFrame columns to numpy arrays
    probabilities = df_COPT['Probability'].values
    available_capacity = df_COPT['available_capacity'].values
    
    # Calculate lost load using numpy operations
    lost_load = probabilities * np.maximum(0, load - available_capacity)
    
    # Return the sum of the lost load
    return lost_load.sum()
def get_LOLP(Cgens, Ugens, P_round, NLh_load, COPT_function):
    """
    Calculate Loss of Load Probability (LOLP).

    Parameters:
    - Cgens (list): List of generator capacities.
    - Ugens (list): List of the unavailabilities of these respective generators.
    - P_round (int): Rounding factor.
    - NLh_load (list): Load values.
    - COPT_function (function): Function to calculate COPT.

    Returns:
    - DataFrame: DataFrame containing hourly LOLP (every row has LOLP in that hour).
    """
    
    # Calculate COPT
    df_COPT = COPT_function(Cgens, Ugens, P_round)

    # Rename the Capacity column to Capacity_Outage
    df_COPT = df_COPT.rename(columns={'Capacity': 'Capacity_Outage'})
    
    # Add column available_capacity which is the total capacity minus the lost capacity
    Ctot_rounded = np.sum(np.multiply(P_round, np.round(np.divide(Cgens, P_round))))
    df_COPT['available_capacity'] = Ctot_rounded - df_COPT['Capacity_Outage']
    
    # Initialize df_LOLP DataFrame
    df_LOLP = pd.DataFrame(columns=['load', 'LOLP'])
    df_LOLP['load'] = NLh_load
    
    # Round load down to nearest P_round (floor)
    df_LOLP['load_rounded'] = np.floor(df_LOLP['load'] / P_round) * P_round
    
    # prune COPT to only have available capacity and cumulative probability columns, drop index
    # df_COPT = df_COPT[['available_capacity', 'Cumulative Probability']]
    df_COPT = df_COPT.reset_index(drop=True)

    df_LOLP = df_LOLP.merge(df_COPT, left_on='load_rounded', right_on='available_capacity', how='left') 
    df_LOLP['LOLP'] = df_LOLP['Cumulative Probability']

    # Calculate cummulative LOLP
    df_LOLP['cummulative_lolp'] = df_LOLP['Cumulative Probability'].cumsum()
    df_LOLP['cummulative_lolp'] = df_LOLP['cummulative_lolp'] / (df_LOLP.index + 1)
    # ADD LOEE FOR EACH HOUR (LOAD)
    df_LOLP['LOEE'] = df_LOLP['load'].apply(lambda x: get_LOEE(x, df_COPT))
    return df_LOLP

# Test case example
df_lolp4gen = get_LOLP([200, 100, 200, 500], [0.05, 0.03, 0.04, 0.06], 100, [550 for i in range(5000)] + [350 for i in range(3760)], COPT_function)
df_lolp4gen['LOEE'].sum()
# Should be about 22 GW

# %%
df_LOLP = get_LOLP(Cgens, Ugens, P_round, NLh_load, COPT_function)
# TOTAL LOLP
# Sum of LOLP divided by number of hours in a year (8760)
total_lolp = df_LOLP['LOLP'].sum()/8760
df_LOLP_copy = df_LOLP.copy()
df_LOLP_copy['LOLP'] = df_LOLP_copy['LOLP']
# Plot just the LOLP against time
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_LOLP_copy.index, y=df_LOLP_copy['LOLP'], name='LOLP', mode='lines', line=dict(color='royalblue', width=4)))
fig.update_layout(title='LOLP vs Time', 
                  title_font_size=30, 
                  xaxis_title='Time (h)', 
                  yaxis_title='Hourly LOLP', 
                  font_size=20, 
                  legend_font_size=20, 
                width=800, height=800)
# Add subtitle of total LOLP
fig.add_annotation(x=1, y=1.0745,
            text="Yearly LOLP: "+str(round(total_lolp,10)),
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=20))
fig.write_image("./plots/LOLP2_reverse.png")
fig.show()
# Reverse it back


# %% [markdown]
# ## Question 6
# ### Consider the first scenario
# + First is base
# + Second is shifted by 24h
# + Third is shifted by 48h
# ### Calculate
# + Total installed capacity of offshore wind
# + Capacity factor of offshore wind
# + Subtract the wind production from the system load and calculate LOLP/LOLE again.

# %%
# Maximum capacity of wind scenario 1
P_wind1 = np.max(P_off_wind1)
print('Maximum capacity of wind scenario 1: ', P_wind1)
# Sum of the capacities of wind scenario 1
P_wind1_sum = np.sum(P_off_wind1)
print('Sum of the capacities of wind scenario 1: ', P_wind1_sum)
# Average generation
P_wind1_avg = np.mean(P_off_wind1)
print('Average generation: ', P_wind1_avg)
# Capacity factor 
P_wind1_cf = P_wind1_avg / P_wind1 # this is the same as (P_wind1_sum / 8760) / P_wind1
P_wind1_cf2 = P_wind1_sum / (P_wind1 * 8760)
print('Capacity factor: ', P_wind1_cf)
print('Capacity factor 2: ', P_wind1_cf2)


# Adjust load by subtracting wind generation
NLh_load_wind1 = np.subtract(NLh_load, P_off_wind1)

# Recalculate LOLP
df_LOLP_wind1 = get_LOLP(Cgens, Ugens, P_round, NLh_load_wind1, COPT_function)

# Calculate total LOLP
total_lolp_wind1 = df_LOLP_wind1['LOLP'].sum()/8760

# Subtract scenario2
NLh_load_wind2 = np.subtract(NLh_load, P_off_wind2)
df_LOLP_wind2 = get_LOLP(Cgens, Ugens, P_round, NLh_load_wind2, COPT_function)
total_lolp_wind2 = df_LOLP_wind2['LOLP'].sum()/8760

# Subtract scenario3
NLh_load_wind3 = np.subtract(NLh_load, P_off_wind3)
df_LOLP_wind3 = get_LOLP(Cgens, Ugens, P_round, NLh_load_wind3, COPT_function)

# Calculate total LOLP
total_lolp_wind3 = df_LOLP_wind3['LOLP'].sum()/8760

# Plot WIND LOLP and only LOLP
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_LOLP.index, y=df_LOLP['LOLP'], name='LOLP no-wind', mode='lines', line=dict(color='royalblue', width=4), opacity=0.8))
fig.add_trace(go.Scatter(x=df_LOLP_wind1.index, y=df_LOLP_wind1['LOLP'], name='LOLP wind adjusted', mode='lines', line=dict(color='firebrick', width=4)))

fig.update_layout(title='LOLP vs Time', 
                  title_font_size=30, 
                  xaxis_title='Time (h)', 
                  yaxis_title='Hourly LOLP', 
                  font_size=20, 
                  legend_font_size=20, 
                width=800, height=800)
# Update legend position
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.05
))
# Remove background of legend
fig.update_layout(legend= {'bgcolor': 'rgba(255,255,255,0.33)'})
# Add subtitle of total LOLP
fig.add_annotation(x=0.3, y=1.0745,
            text="Yearly adjusted LOLP: "+str(round(total_lolp_wind1,10)),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xref="paper",
            yref="paper",
            font=dict(size=20))
            
fig.add_annotation(x=0.3, y=1.0745,
            text="Yearly no-wind LOLP: "+str(round(total_lolp,10)),
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=20))

fig.write_image("./plots/LOLP3.png")
fig.show()


# %% [markdown]
# ## Question 7
# ### Calculate
# + capacity credit of wind energy / load carrying capability
# - in MW (Effective Load Carrying Capability)
# - in % of peak load
# + Does it make a difference if the other wind scenarios (i.e. the
# scenarios that are 24 h and 48 h shifted in time) are used, and why? And how can we avoid this?

# %%
# Print total LOLP and total LOLP wind
print('Total LOLP: ', total_lolp)
print('LOLE', total_lolp*8760)
print('Total LOLP wind: ', total_lolp_wind1)
# Peak load
peak_load = np.max(NLh_load)
print('Peak load: ', peak_load)

# %%
def plot_wind_energy_pdf(P_off_wind1, P_round):
    """
    Plot the probability density function of wind energy data using a specified bin size.

    Parameters:
    - P_off_wind1 (list): List of generated wind energy for every hour of the year.
    - P_round (int): Bin size for the histogram.
    """
    
    # Calculate histogram counts
    hist_counts, bin_edges = np.histogram(P_off_wind1, bins=np.arange(0, np.max(P_off_wind1) + P_round, P_round))
    
    # Normalize the counts
    hist_normalized = hist_counts / (len(P_off_wind1) * P_round)
    
    # Mid-point of each bin
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Convert to DataFrame
    df_wind_energy_pdf = pd.DataFrame({'Wind Energy (MW)': bin_mids, 'Probability Density': hist_normalized})

    # Plot using plotly.graph_objects
    fig = go.Figure(data=[go.Bar(x=bin_mids, y=hist_normalized, width=P_round)])
    fig.update_layout(title="Wind Energy Probability Density Function",
                      xaxis_title="Wind Energy (MW)",
                      yaxis_title="Probability Density",
                      bargap=0.01)
    fig.update_layout(title="Wind Energy Probability Density Function",
                  xaxis_title="Wind Energy (MW)",
                  yaxis_title="Probability Density",
                  bargap=0.01)
    # Size 800x800
    fig.update_layout(width=800, height=800)
    # Title font size 30, Axis font size 20, Legend font size 20
    fig.update_layout(title_font_size=30, font_size=20, legend_font_size=20)
    return fig, df_wind_energy_pdf

# %%
# We can avoid by representing the wind farm as a single wind generator with a probability density function of generation
fig, df_wind_energy_pdf = plot_wind_energy_pdf(P_off_wind1, P_round)
fig.write_image("./plots/wind_energy_pdf.png")
fig.show()

# %%
def get_yearly_LOLP(Cgens, Ugens, NLh_load, COPT_function,P_round = 50):
    # Calculate LOLP
    df_LOLP = get_LOLP(Cgens, Ugens, P_round, NLh_load, COPT_function)
    # Calculate yearly LOLP
    yearly_LOLP = df_LOLP['LOLP'].sum()/8760
    return yearly_LOLP

def get_yearly_LOEE(Cgens, Ugens, NLh_load, COPT_function, P_round = 50):
    df_LOLP = get_LOLP(Cgens, Ugens, P_round, NLh_load, COPT_function)
    # Calculate yearly LOEE
    yearly_LOEE = df_LOLP['LOEE'].sum()
    return yearly_LOEE

# Recalculator of LOLP
def recalculate_LOLP(Cgens, Ugens, NLh_load, COPT_function, wind_scenario, P_round = 50):
    fig, df_wind_energy_pdf = plot_wind_energy_pdf(P_off_wind1, P_round)
    # df_wind_energy_pdf has columns 'Wind Energy (MW)' and 'Probability Density'

    peak_load = np.max(NLh_load)
    # Transform to generator representation (transform density to actual probabilities of generation)
    df_wind_energy_pdf['Probability Density'] = df_wind_energy_pdf['Probability Density'] * P_round
    # Rename 'Probability Density' to Probability
    df_wind_energy_pdf = df_wind_energy_pdf.rename(columns={'Probability Density': 'Probability'})

    # Add LOLP column to df_wind_energy_pdf
    # Subtract wind energy from load for calculation
    # df_wind_energy_pdf['Yearly_LOLP'] = get_yearly_LOLP(Cgens, Ugens, np.subtract(NLh_load, df_wind_energy_pdf['Wind Energy (MW)']), COPT_function, P_round)
    df_wind_energy_pdf['Yearly_LOLP'] = df_wind_energy_pdf['Wind Energy (MW)'].apply(lambda wind_energy: get_yearly_LOLP(Cgens, Ugens, np.subtract(NLh_load, wind_energy), COPT_function, P_round))
    df_wind_energy_pdf['Yearly_LOEE'] = df_wind_energy_pdf['Wind Energy (MW)'].apply(lambda wind_energy: get_yearly_LOEE(Cgens, Ugens, np.subtract(NLh_load, wind_energy), COPT_function, P_round))
    # The column 'Probability' is the weight
    weighted_total_lolp_wind = np.sum(np.multiply(df_wind_energy_pdf['Probability'], df_wind_energy_pdf['Yearly_LOLP']))
    weighted_total_LOEE_wind = np.sum(np.multiply(df_wind_energy_pdf['Probability'], df_wind_energy_pdf['Yearly_LOEE']))

    return weighted_total_LOEE_wind,weighted_total_lolp_wind, df_wind_energy_pdf

# Test case
print(get_yearly_LOLP([200, 100, 200, 500], [0.05, 0.03, 0.04, 0.06],[500 for i in range(5000)] + [350 for i in range(3760)] , COPT_function, 100))

# %%
# Get curve for the first scenario
total_loee_wind1, total_lolp_wind1, df_wind_scenario1 = recalculate_LOLP(Cgens, Ugens, NLh_load, COPT_function, P_off_wind1, P_round)
print("LOLP WIND1:", total_lolp_wind1)
print("LOEE WIND1:", total_loee_wind1)

# Since the scenarios are the same we can just copy the values
total_loee_wind2, total_lolp_wind2, df_wind_scenario2 = total_loee_wind3, total_lolp_wind3, df_wind_scenario3 = total_loee_wind1, total_lolp_wind1, df_wind_scenario1


# Second
# total_loee_wind2,total_lolp_wind2, df_wind_scenario2 = recalculate_LOLP(Cgens, Ugens, NLh_load, COPT_function, P_off_wind2, P_round)
print("LOLP WIND2:", total_lolp_wind2)
print("LOEE WIND2:", total_loee_wind2)
# Third
# total_loee_wind3,total_lolp_wind3, df_wind_scenario3 = recalculate_LOLP(Cgens, Ugens, NLh_load, COPT_function, P_off_wind3, P_round)
print("LOLP WIND3:", total_lolp_wind3)
print("LOEE WIND3:", total_loee_wind3)

# Total lolps are the same
# Plot wind1 scenario Yearly_LOLP vs Wind Energy (MW)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_wind_scenario1['Wind Energy (MW)'], y=df_wind_scenario1['Yearly_LOLP'], name='LOLP', mode='lines', line=dict(color='royalblue', width=4)))
fig.update_layout(title='Yearly LOLP vs Wind Energy (MW)')



# Update size 800x800 title font size 30, rest 20
fig.update_layout(width=800, height=800, title_font_size=30, font_size=20)

fig.update_layout(title='Yearly LOLP vs Wind Energy (MW)', 
                  title_font_size=30, 
                  xaxis_title='Wind Energy (MW)', 
                  yaxis_title='Yearly LOLP', 
                  font_size=20, 
                  legend_font_size=20, 
                width=800, height=800)
# Update legend position
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.05
))
# Add subtitle of total LOLP

# Remove background of legend
fig.update_layout(legend= {'bgcolor': 'rgba(255,255,255,0.33)'})
fig.write_image("./plots/LOLPQ7.png")
fig.show()

# %%
# GET ELCC AND CAPACITY CREDIT
LOEE_NOWIND = get_yearly_LOEE(Cgens, Ugens, NLh_load, COPT_function, P_round)
print("No wind LOEE:", LOEE_NOWIND)
print("Wind LOEE:", total_loee_wind1)
ELCC = LOEE_NOWIND - total_loee_wind1
print("ELCC: ", ELCC)
print(peak_load)
CAPACITY_CREDIT = 100*ELCC / peak_load
print(f"Capacity credit: {'%.2f' % CAPACITY_CREDIT} %")


