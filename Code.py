import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel("Kickstarter.xlsx")


##################### Part 1: Model Building #########################
###### Pre-Processing ###########
# Extract timelines
df['campaign_duration'] = (df['deadline'] - df['launched_at']).dt.days
df['preparation_duration'] = (df['launched_at'] - df['created_at']).dt.days


# Drop columns
df = df.drop([
    'id', 'name', 'pledged', 'state_changed_at', 'staff_pick', 'backers_count', 
    'static_usd_rate', 'usd_pledged', 'spotlight', 'name_len', 'blurb_len', 
    'state_changed_at_weekday', 'state_changed_at_month', 'state_changed_at_day', 
    'state_changed_at_yr', 'state_changed_at_hr', 'staff_pick', 
    'deadline', 'created_at', 'launched_at'], axis=1)

# Filter rows for only successful or failed state
df = df[df['state'].isin(['successful', 'failed'])]

# Check skewness for the goal column
skewness_goal = df['goal'].skew()
print(f"Skewness of 'goal': {skewness_goal}")

# Apply log transformation if skewness is high
if abs(skewness_goal) > 1:
    df['goal'] = np.log1p(df['goal'])
    print("Log transformation applied to 'goal'.")
    

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, columns=[
    'state', 'country', 'currency', 'category', 
    'deadline_weekday', 'created_at_weekday', 
    'launched_at_weekday', 'main_category'], drop_first=True)

# Split features and target
X = df.drop(['state_successful'], axis=1)
y = df['state_successful']

########## Keep seperate dataset for clustering    
clustering_data = X.copy()


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

###### Outlier Removal Using Isolation Forest ###########
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso.fit_predict(X_scaled)

# Identify and remove outliers (label = -1)
X_filtered = X_scaled[outlier_labels == 1]
y_filtered = y[outlier_labels == 1]

# Debug: Check the size of filtered data
print(f"Remaining samples after Isolation Forest: {X_filtered.shape[0]}")

###### Handle Imbalance Using SMOTE ###########
# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Debug: Check the size of resampled data
print(f"Samples after SMOTE: {X_resampled.shape[0]}")

###### Train-Test Split ###########
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=0)

###### Model Building ###########
# Train Gradient Boosting model
gb = GradientBoostingClassifier(
    n_estimators=500,      
    max_depth=4,           
    random_state=42        
)
model_gb = gb.fit(X_train, y_train)

# Predictions and metrics
y_test_pred = model_gb.predict(X_test)
precision_gb = precision_score(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
recall_gb = recall_score(y_test, y_test_pred)
f1_gb = f1_score(y_test, y_test_pred)
print(f"Accuracy Score: {accuracy}")
print(f"Precision Score: {precision_gb}")
print(f"Recall Score: {recall_gb}")
print(f"F1 Score: {f1_gb}")

#Accuracy Score: 0.8296169239565466
#Precision Score: 0.8395204949729311
#Recall Score: 0.8192452830188679
#F1 Score: 0.8292589763177999


####### Cross Validation #################
# Define KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(gb, X_resampled, y_resampled, cv=kfold, scoring='accuracy')

# Print cross-validation scores
print("Cross-validation scores for each fold: ", cv_scores)
print("Mean accuracy: ", np.mean(cv_scores))
print("Standard deviation of accuracy: ", np.std(cv_scores))



############ Check for overfitting #################
y_train_pred = model_gb.predict(X_train)
precision_gb_of = precision_score(y_train, y_train_pred)
accuracy_of = accuracy_score(y_train, y_train_pred)
recall_gb_of = recall_score(y_train, y_train_pred)
f1_gb_of = f1_score(y_train, y_train_pred)
print(f"Accuracy Score: {accuracy_of}")
print(f"Precision Score: {precision_gb_of}")
print(f"Recall Score: {recall_gb_of}")
print(f"F1 Score: {f1_gb_of}")


#Accuracy Score: 0.9039707124753591
#Precision Score: 0.9146790769827419
#Recall Score: 0.89
#F1 Score: 0.9021707946829874



################ Model Grading #######################################

df_grading = pd.read_excel("Kickstarter-Grading.xlsx")


###### Pre-Processing ###########
# Extract timelines
df_grading['campaign_duration'] = (df_grading['deadline'] - df_grading['launched_at']).dt.days
df_grading['preparation_duration'] = (df_grading['launched_at'] - df_grading['created_at']).dt.days


# Drop columns
df_grading = df_grading.drop([
    'id', 'name', 'pledged', 'state_changed_at', 'staff_pick', 'backers_count', 
    'static_usd_rate', 'usd_pledged', 'spotlight', 'name_len', 'blurb_len', 
    'state_changed_at_weekday', 'state_changed_at_month', 'state_changed_at_day', 
    'state_changed_at_yr', 'state_changed_at_hr', 'staff_pick', 
    'deadline', 'created_at', 'launched_at'], axis=1)

# Filter rows for only successful or failed state
df_grading = df_grading[df_grading['state'].isin(['successful', 'failed'])]

# Check skewness for the goal column
skewness_goal = df_grading['goal'].skew()
print(f"Skewness of 'goal': {skewness_goal}")

# Apply log transformation if skewness is high
if abs(skewness_goal) > 1:
    df_grading['goal'] = np.log1p(df_grading['goal'])
    print("Log transformation applied to 'goal'.")
    

# Convert categorical columns to dummy variables
df_grading = pd.get_dummies(df_grading, columns=[
    'state', 'country', 'currency', 'category', 
    'deadline_weekday', 'created_at_weekday', 
    'launched_at_weekday', 'main_category'], drop_first=True)

# Split features and target
X_grading = df_grading.drop(['state_successful'], axis=1)
y_grading = df_grading['state_successful']



# Standardize features
scaler = StandardScaler()
X_grading = scaler.fit_transform(X_grading)

###### Outlier Removal Using Isolation Forest ###########
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso.fit_predict(X_grading)

# Identify and remove outliers (label = -1)
X_grading = X_grading[outlier_labels == 1]
y_grading = y_grading[outlier_labels == 1]

# Debug: Check the size of filtered data
print(f"Remaining samples after Isolation Forest: {X_grading.shape[0]}")

###### Handle Imbalance Using SMOTE ###########
# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_grading, y_grading = smote.fit_resample(X_grading, y_grading)

# Debug: Check the size of resampled data
print(f"Samples after SMOTE: {X_grading.shape[0]}")



###### SCORING AND PREDICTIONS ###################

y_graded_pred = model_gb.predict(X_grading)
precision_graded = precision_score(y_grading, y_graded_pred)
accuracy_graded = accuracy_score(y_grading, y_graded_pred)
recall_graded = recall_score(y_grading, y_graded_pred)
f1_graded = f1_score(y_grading, y_graded_pred)
print(f"Accuracy Score: {accuracy_graded}")
print(f"Precision Score: {precision_graded}")
print(f"Recall Score: {recall_graded}")
print(f"F1 Score: {f1_graded}")






############################## Part 2: Clustering #########################################

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

########################## Preprocessing ##########################


iso = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso.fit_predict(clustering_data)
clustering_data_filtered = clustering_data[outlier_labels == 1]  # Keep only non-outliers
print(f"Number of outliers removed: {sum(outlier_labels == -1)}")
print(f"Shape after removing outliers: {clustering_data_filtered.shape}")

# Step 3: Scale data
scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data_filtered)

########################## Determine Optimal Number of Clusters ##########################

# Plot a dendrogram for hierarchical clustering
plt.figure(figsize=(10, 8))
linkage_matrix = linkage(clustering_data_scaled, method='ward')  # Ward linkage minimizes variance
dendrogram(linkage_matrix)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Elbow method to find optimal number of clusters
wcss = []
n_clusters_range = range(2, 11)

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(clustering_data_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(n_clusters_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method: Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.tight_layout()
plt.show()

########################## Clustering ##########################

# Use PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
clustering_data_pca = pca.fit_transform(clustering_data_scaled)

# Perform KMeans clustering
optimal_clusters = 3  # Replace with chosen number from elbow/dendrogram analysis
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(clustering_data_pca)

# Add clusters to the dataset
clustering_data_filtered['Cluster'] = clusters

# Evaluate clustering performance
silhouette_avg = silhouette_score(clustering_data_pca, clusters)
print(f"Silhouette Score for {optimal_clusters} Clusters: {silhouette_avg:.2f}")

# Visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=clustering_data_pca[:, 0],
    y=clustering_data_pca[:, 1],
    hue=clusters,
    palette="Set2",
    legend="full"
)
plt.title("Clusters of Kickstarter Projects (PCA-Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

######################## Cluster Analysis ########################

# Important features for analysis
important_features = [
    'preparation_duration', 'campaign_duration', 'goal',
    'video', 'show_feature_image', 'category_Documentary'
]

# Group data by clusters and calculate statistics
cluster_summary = clustering_data_filtered.groupby('Cluster')[important_features].agg(['mean', 'median', 'std'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("\nCluster Characteristics:")
print(cluster_summary)


"""
Cluster Characteristics:
        preparation_duration                    campaign_duration         \
                        mean median         std              mean median   
Cluster                                                                    
0                  95.655402   20.0  287.268247         32.485007   30.0   
1                  54.605714   13.0  142.970969         33.637438   30.0   
2                  61.727716   12.0  209.435023         33.874295   30.0   

                        goal                         video                   \
               std      mean    median       std      mean median       std   
Cluster                                                                       
0        11.358294  8.668866  8.779711  1.636499  0.865302    1.0  0.341482   
1        12.707721  8.536533  8.517393  2.009739  0.661478    1.0  0.473254   
2        13.426226  8.418540  8.517393  1.729876  0.577175    1.0  0.494046   

        show_feature_image                  category_Documentary         \
                      mean median       std                 mean median   
Cluster                                                                   
0                 0.048548    0.0  0.214973             0.025702    0.0   
1                 0.025616    0.0  0.158002             0.016158    0.0   
2                 0.028188    0.0  0.165523             0.018284    0.0   

                   
              std  
Cluster            
0        0.158283  
1        0.126094  
2        0.133988  
"""





























