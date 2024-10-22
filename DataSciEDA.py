import pandas as pd
df = pd.read_feather('nc_reduced_data.feather')
print(df['file_source'].unique())

# Data Dividing
durham_data = df[df['file_source'] == 'nc_durham']
statewide_data = df[df['file_source'] == 'nc_statewide']
print(len(durham_data))

# EDA for Durham
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

# Assuming black population is 38%
def race_proportion(df):
    df.loc[:, 'subject_race_black'] = df['subject_race_cat'].apply(lambda x: 'Black' if x == 'black' else 'All Population')
    return df

# Apply race proportion to both datasets
durham_data = race_proportion(durham_data)
statewide_data = race_proportion(statewide_data)

# --- Per Capita Calculations for Arrests and Stops ---
durham_population = 326024
statewide_population = 10480000

# Calculate per capita arrests
durham_arrests_per_capita = len(durham_data[durham_data['outcome_cat'] == 'arrest']) / durham_population
statewide_arrests_per_capita = len(statewide_data[statewide_data['outcome_cat'] == 'arrest']) / statewide_population

# --- Plot Per Capita Comparisons ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Arrests per capita
sns.barplot(x=['Durham', 'Statewide'], y=[durham_arrests_per_capita, statewide_arrests_per_capita], ax=ax[0])
ax[0].set_title('Arrests Per Capita')
ax[0].set_ylabel('Arrests Per Person')

plt.tight_layout()  # Ensure layout fits well
plt.savefig('14.png')
plt.show()

# Save similar plots using state sample data
sample_size = len(durham_data)
statewide_sample = statewide_data.sample(n=sample_size, random_state=42)

# Calculate per capita arrests/stops with sample
statewide_arrests_sample_per_capita = len(statewide_sample[statewide_sample['outcome_cat'] == 'arrest']) / statewide_population
statewide_stops_sample_per_capita = len(statewide_sample) / statewide_population

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Arrests per capita (with sample)
sns.barplot(x=['Durham', 'Statewide Sample'], y=[durham_arrests_per_capita, statewide_arrests_sample_per_capita], ax=ax[0])
ax[0].set_title('Arrests Per Capita (Sample)')
ax[0].set_ylabel('Arrests Per Person')


plt.tight_layout()
plt.savefig('14s.png')

# Set plot style
plt.figure(figsize=(10, 6))

# Plot 1: Distribution of arrests by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Arrest Distribution by Race (Durham)')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("1.png")

# Plot 2: Distribution of citations issued by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Citation Distribution by Race (Durham)')
plt.xlabel('Citation Issued')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("2.png")
# Plot 3: Age distribution by race (Durham)
plt.figure()
sns.histplot(data=durham_data, x='subject_age', hue='subject_race_black', kde=True, bins=30)
plt.title('Age Distribution by Race (Durham)')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("3.png")
# Plot 4: Frequency of warnings by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Warning Issued by Race (Durham)')
plt.xlabel('Warning Issued')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("4.png")
# Plot 5: Distribution of search conducted by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Search Conducted by Race (Durham)')
plt.xlabel('Search Conducted')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("5.png")
# Plot 6: Frisk performed by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Frisk Performed by Race (Durham)')
plt.xlabel('Frisk Performed')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("6.png")
# Plot 7: Age vs. Citation Issued colored by race (Durham)
plt.figure()
sns.boxplot(data=durham_data, x='outcome_cat', y='subject_age', hue='subject_race_black')
plt.title('Age vs. Citation Issued by Race (Durham)')
plt.xlabel('Citation Issued')
plt.ylabel('Age')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("7.png")
# Plot 8: Search Person by Race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Search Person by Race (Durham)')
plt.xlabel('Search Person')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("8.png")
# Plot 9: Outcome of Stops by Race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Outcome of Stops by Race (Durham)')
plt.xlabel('Type of Stop')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("9.png")
# Plot 10: Reason for Stop by Race (Durham)
plt.figure(figsize=(12, 6))
top_reasons = durham_data['reason_for_stop_cat'].value_counts().nlargest(5).index
filtered_data = durham_data[durham_data['reason_for_stop_cat'].isin(top_reasons)]
sns.countplot(data=filtered_data, x='reason_for_stop_cat', hue='subject_race_black')
plt.title('Top Reasons for Stop by Race (Durham)')
plt.xlabel('Reason for Stop')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.xticks(rotation=45)
plt.show()
plt.tight_layout()
plt.savefig("10.png")
# Define severity classification function
def classify_severity(reason):
    if pd.isnull(reason):
        return 'unknown'
    elif 'speed' in reason.lower() or 'seatbelt' in reason.lower():
        return 'low'
    elif 'stop' in reason.lower() or 'impaired' in reason.lower() or 'vehicle' in reason.lower():
        return 'medium'
    elif 'safe' in reason.lower() or 'license' in reason.lower():
        return 'high'
    else:
        return 'unknown'

# Apply severity classification
durham_data['severity_level'] = durham_data['reason_for_stop_cat'].apply(classify_severity)

# Plot 1: Distribution of severity levels by race (Durham)
plt.figure(figsize=(10, 6))
sns.countplot(data=durham_data, x='severity_level', hue='subject_race_black')
plt.title('Distribution of Severity Levels by Race (Durham)')
plt.xlabel('Severity Level')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("11.png")
# Plot 2: Crime severity percentage breakdown by race (Durham)
plt.figure(figsize=(10, 6))
severity_percentage = durham_data.groupby('subject_race_black')['severity_level'].value_counts(normalize=True).unstack()
severity_percentage.plot(kind='bar', stacked=True, colormap="viridis", figsize=(10,6))
plt.title('Percentage of Crime Severity Levels by Race (Durham)')
plt.xlabel('Race')
plt.ylabel('Percentage')
plt.legend(title='Severity Level')
plt.show()
plt.tight_layout()
plt.savefig("12.png")
# Plot 3: Average age of subjects in each severity level by race (Durham)
plt.figure(figsize=(10, 6))
sns.boxplot(data=durham_data, x='severity_level', y='subject_age', hue='subject_race_black')
plt.title('Age Distribution Across Severity Levels by Race (Durham)')
plt.xlabel('Severity Level')
plt.ylabel('Age')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("13.png")

# before working on state related data, acknowledge that it is too large.
# We take sample of it that is the same size as the durham data, to enable fair contrast.

sample_size = len(durham_data)
statewide_sample = statewide_data.sample(n=sample_size, random_state=42)

sav = durham_data
durham_data = statewide_sample

# Plot 1: Distribution of arrests by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Arrest Distribution by Race (Durham)')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("1s.png")

# Plot 2: Distribution of citations issued by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Citation Distribution by Race (Durham)')
plt.xlabel('Citation Issued')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("2s.png")
# Plot 3: Age distribution by race (Durham)
plt.figure()
sns.histplot(data=durham_data, x='subject_age', hue='subject_race_black', kde=True, bins=30)
plt.title('Age Distribution by Race (Durham)')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("3s.png")
# Plot 4: Frequency of warnings by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Warning Issued by Race (Durham)')
plt.xlabel('Warning Issued')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("4s.png")
# Plot 5: Distribution of search conducted by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Search Conducted by Race (Durham)')
plt.xlabel('Search Conducted')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("5s.png")
# Plot 6: Frisk performed by race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Frisk Performed by Race (Durham)')
plt.xlabel('Frisk Performed')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("6s.png")
# Plot 7: Age vs. Citation Issued colored by race (Durham)
plt.figure()
sns.boxplot(data=durham_data, x='outcome_cat', y='subject_age', hue='subject_race_black')
plt.title('Age vs. Citation Issued by Race (Durham)')
plt.xlabel('Citation Issued')
plt.ylabel('Age')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("7s.png")
# Plot 8: Search Person by Race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Search Person by Race (Durham)')
plt.xlabel('Search Person')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("8s.png")
# Plot 9: Outcome of Stops by Race (Durham)
plt.figure()
sns.countplot(data=durham_data, x='outcome_cat', hue='subject_race_black')
plt.title('Outcome of Stops by Race (Durham)')
plt.xlabel('Type of Stop')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("9s.png")
# Plot 10: Reason for Stop by Race (Durham)
plt.figure(figsize=(12, 6))
top_reasons = durham_data['reason_for_stop_cat'].value_counts().nlargest(5).index
filtered_data = durham_data[durham_data['reason_for_stop_cat'].isin(top_reasons)]
sns.countplot(data=filtered_data, x='reason_for_stop_cat', hue='subject_race_black')
plt.title('Top Reasons for Stop by Race (Durham)')
plt.xlabel('Reason for Stop')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.xticks(rotation=45)
plt.show()
plt.tight_layout()
plt.savefig("10s.png")
# Define severity classification function
def classify_severity(reason):
    if pd.isnull(reason):
        return 'unknown'
    elif 'speed' in reason.lower() or 'seatbelt' in reason.lower():
        return 'low'
    elif 'stop' in reason.lower() or 'impaired' in reason.lower() or 'vehicle' in reason.lower():
        return 'medium'
    elif 'safe' in reason.lower() or 'license' in reason.lower():
        return 'high'
    else:
        return 'unknown'

# Apply severity classification
durham_data['severity_level'] = durham_data['reason_for_stop_cat'].apply(classify_severity)

# Plot 1: Distribution of severity levels by race (State)
plt.figure(figsize=(10, 6))
sns.countplot(data=durham_data, x='severity_level', hue='subject_race_black')
plt.title('Distribution of Severity Levels by Race (State)')
plt.xlabel('Severity Level')
plt.ylabel('Count')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("11s.png")
# Plot 2: Crime severity percentage breakdown by race (State)
plt.figure(figsize=(10, 6))
severity_percentage = durham_data.groupby('subject_race_black')['severity_level'].value_counts(normalize=True).unstack()
severity_percentage.plot(kind='bar', stacked=True, colormap="viridis", figsize=(10,6))
plt.title('Percentage of Crime Severity Levels by Race (State)')
plt.xlabel('Race')
plt.ylabel('Percentage')
plt.legend(title='Severity Level')
plt.show()
plt.tight_layout()
plt.savefig("12s.png")
# Plot 3: Average age of subjects in each severity level by race (Durham)
plt.figure(figsize=(10, 6))
sns.boxplot(data=durham_data, x='severity_level', y='subject_age', hue='subject_race_black')
plt.title('Age Distribution Across Severity Levels by Race (Durham)')
plt.xlabel('Severity Level')
plt.ylabel('Age')
plt.legend(title='Race', labels=['All Population', 'Black'])
plt.show()
plt.tight_layout()
plt.savefig("13s.png")

durham_data = sav

# Convert datetime column
durham_data['datetime'] = pd.to_datetime(durham_data['datetime'], errors='coerce')
# Over here, I converted state data to the samples. Otherwise it will run forever.
statewide_data = statewide_sample
statewide_data['datetime'] = pd.to_datetime(statewide_data['datetime'], errors='coerce')

# Plot 1: Distribution of events over time (yearly)
plt.figure(figsize=(12, 6))
durham_data['year'] = durham_data['datetime'].dt.year
statewide_data['year'] = statewide_data['datetime'].dt.year
sns.histplot(data=durham_data, x='year', color='blue', label='Durham', kde=True, bins=20)
sns.histplot(data=statewide_data, x='year', color='orange', label='Statewide', kde=True, bins=20, alpha=0.6)
plt.title('Yearly Distribution of Events: Durham vs. Statewide')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()
plt.savefig("c1.png")
# Plot 2: Time of day distribution of events
plt.figure(figsize=(12, 6))
durham_data['hour'] = durham_data['datetime'].dt.hour
statewide_data['hour'] = statewide_data['datetime'].dt.hour
sns.histplot(data=durham_data, x='hour', color='blue', label='Durham', kde=True, bins=24)
sns.histplot(data=statewide_data, x='hour', color='orange', label='Statewide', kde=True, bins=24, alpha=0.6)
plt.title('Time of Day Distribution: Durham vs. Statewide')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.legend()
plt.show()
plt.savefig("c2.png")
# Plot 3: Monthly trend of events
plt.figure(figsize=(12, 6))
durham_data['month'] = durham_data['datetime'].dt.month
statewide_data['month'] = statewide_data['datetime'].dt.month
sns.histplot(data=durham_data, x='month', color='blue', label='Durham', kde=True, bins=12)
sns.histplot(data=statewide_data, x='month', color='orange', label='Statewide', kde=True, bins=12, alpha=0.6)
plt.title('Monthly Trend of Events: Durham vs. Statewide')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend()
plt.show()
plt.savefig("c3.png")
# Plot 4: Yearly trend by race (focusing on Black population)
plt.figure(figsize=(12, 6))
black_durham_data = durham_data[durham_data['subject_race_cat'] == 'black']
black_statewide_data = statewide_data[statewide_data['subject_race_cat'] == 'black']
sns.histplot(data=black_durham_data, x='year', color='blue', label='Durham (Black)', kde=True, bins=20)
sns.histplot(data=black_statewide_data, x='year', color='orange', label='Statewide (Black)', kde=True, bins=20, alpha=0.6)
plt.title('Yearly Trend of Events for Black Population: Durham vs. Statewide')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()
plt.savefig("c4.png")