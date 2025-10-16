# Import required Python libraries for data analysis and visualisation
import pandas as pd              # pandas is used for loading and handling tabular data
import matplotlib.pyplot as plt  # matplotlib is used for creating plots and figures
import seaborn as sns            # seaborn is used to make statistical plots like heatmaps

# Step 1: Load the dataset
# The CSV file should be in the same directory as this script, or provide the correct path.
# This dataset contains behavioural data such as risk, reward, timing, etc.
df = pd.read_csv("dataset1(1).csv")

# Step 2: Select only numeric columns
# Correlation requires numeric data types. This filters the DataFrame to include only numeric columns
# (e.g., integers and floats), excluding text or categorical variables.
numeric_df = df.select_dtypes(include=["number"])

# Step 3: Compute the correlation matrix
# The .corr() function calculates pairwise correlation coefficients between numeric columns.
# Correlation values range from -1 (strong negative correlation) to +1 (strong positive correlation).
corr = numeric_df.corr()

# Step 4: Create and customise the correlation heatmap
# A heatmap is a useful visual tool to quickly identify strong or weak relationships between variables.
plt.figure(figsize=(10, 8))  # Set the figure size for readability

# Create the heatmap
# - annot=True: display correlation values inside the heatmap cells
# - cmap="coolwarm": use a diverging colour scheme to show negative and positive correlations clearly
# - fmt=".2f": format correlation values to two decimal places
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

# Add a title for clarity
plt.title("Correlation Heatmap of Key Variables")

# Adjust the layout so labels and titles are not cut off
plt.tight_layout()

# Step 5: Save the figure
# This saves the generated heatmap image to the 'figures' folder.
# Ensure that the folder exists before running the script.
plt.savefig("figures/correlation_heatmap.png")

# Step 6: Display the plot
# This renders the plot on the screen, useful for quick inspection during analysis.
plt.show()
