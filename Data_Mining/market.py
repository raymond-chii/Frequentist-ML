import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# Load and prepare data
df = pd.read_csv('Spotify_Youtube.csv')
song_cnt = pd.DataFrame({"Songs":[df["Uri"].nunique()], "Artists":[df["Url_spotify"].nunique()]})


## Top 10 most Streamed song
df.groupby(by = ["Track"], as_index = False)["Stream"].max().sort_values(by = ["Stream"], ascending=False).head(10)


## Top 10 most viewed youtube video
Channel_wise_views = df.groupby(by = ["Track","Channel"], as_index=False)["Views"].max()
Track_Wise_views = df.groupby(by = ["Track"], as_index=False)["Views"].sum()
Track_Wise_views.groupby(by = ["Track"], as_index = False)["Views"].max().sort_values(by = ["Views"], ascending=False).head(10)

# data cleaning

url_cols = ['Unnamed: 0', 'Url_spotify', 'Uri', 'Url_youtube', 'Title', 'Description']
df.drop(url_cols, axis=1, inplace=True)
df.dropna(inplace=True)
df.isna().sum()

df.columns
df.shape

import matplotlib.pyplot as plt

# For Album_type (since it should have fewer categories)
plt.figure(figsize=(8, 6))
df['Album_type'].value_counts().plot(kind='bar')
plt.title('Distribution of Album Types')
plt.tight_layout()
plt.show()

# For others, show value counts
for col in ['Artist', 'Track', 'Album']:
    print(f"\nTop 10 Most Frequent {col}:")
    print(df[col].value_counts().head(10))

import seaborn as sns

# Create subplots for numerical columns
numerical_cols = ['Danceability', 'Energy',
       'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness',
       'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views',
       'Likes', 'Comments', 'Stream']

fig, axes = plt.subplots(6, 3, figsize=(20, 25))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
   sns.histplot(data=df, x=col, ax=axes[idx])
   axes[idx].set_title(f'Distribution of {col}')
   axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

stats_df = pd.DataFrame()

# Calculate statistics for each column
for col in numerical_cols:
   stats = df[col].describe()
   stats_df[col] = stats

# Display the transposed table for better readability
print(stats_df.transpose())

from sklearn.preprocessing import KBinsDiscretizer

# Initialize
kbd = KBinsDiscretizer(n_bins=4, encode='onehot-dense', strategy='quantile')

# Make sure numerical_cols only has numeric data
numeric_data = df[numerical_cols].select_dtypes(include=['float64', 'int64'])
numerical_cols_clean = numeric_data.columns.tolist()

# Transform
encoded = kbd.fit_transform(df[numerical_cols_clean])

# Create column names
col_names = [f'{col}_{i}' for col in numerical_cols_clean for i in range(4)]

# Create DataFrame
binary_df = pd.DataFrame(encoded, columns=col_names[:encoded.shape[1]])

binary_df.head()
binary_df.shape
binary_df.columns

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def analyze_musical_characteristics(rules, min_lift=1.5):

    musical_features = ['Danceability_0', 'Danceability_1', 'Danceability_2', 'Danceability_3',
       'Energy_0', 'Energy_1', 'Energy_2', 'Energy_3', 'Key_0', 'Key_1',
       'Key_2', 'Key_3', 'Loudness_0', 'Loudness_1', 'Loudness_2',
       'Loudness_3', 'Speechiness_0', 'Speechiness_1', 'Speechiness_2',
       'Speechiness_3', 'Acousticness_0', 'Acousticness_1', 'Acousticness_2',
       'Acousticness_3', 'Instrumentalness_0', 'Instrumentalness_1',
       'Instrumentalness_2', 'Instrumentalness_3', 'Liveness_0', 'Liveness_1',
       'Liveness_2', 'Liveness_3', 'Valence_0', 'Valence_1', 'Valence_2',
       'Valence_3', 'Tempo_0', 'Tempo_1', 'Tempo_2', 'Tempo_3']
    
    musical_rules = rules[
        rules['antecedents'].apply(lambda x: any(f in str(x) for f in musical_features)) &
        rules['consequents'].apply(lambda x: any(f in str(x) for f in musical_features)) &
        (rules['lift'] >= min_lift)
    ]
    
    return musical_rules.sort_values('lift', ascending=False)

def analyze_engagement_patterns(rules, min_lift=1.5):

    engagement_metrics = ['Views_0', 'Views_1', 'Views_2', 'Views_3', 'Likes_0', 'Likes_1',
       'Likes_2', 'Likes_3', 'Comments_0', 'Comments_1', 'Comments_2',
       'Comments_3', 'Stream_0', 'Stream_1', 'Stream_2']
    
    # Find rules where musical features lead to engagement
    engagement_rules = rules[
        rules['antecedents'].apply(lambda x: any(f not in str(x) for f in engagement_metrics)) &
        rules['consequents'].apply(lambda x: any(f in str(x) for f in engagement_metrics)) &
        (rules['lift'] >= min_lift)
    ]
    
    return engagement_rules.sort_values('lift', ascending=False)

def analyze_feature_combinations(frequent_itemsets):

    # Look at itemsets with 2 or more items
    complex_patterns = frequent_itemsets[
        frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)
    ]
    
    return complex_patterns.sort_values('support', ascending=False)

def generate_musical_insights(binary_df):

    # Generate frequent itemsets and rules
    frequent_itemsets = apriori(binary_df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets,num_itemsets=binary_df.shape[0], metric='confidence', min_threshold=0.3)
    
    insights = {
        'musical_patterns': analyze_musical_characteristics(rules),
        'engagement_patterns': analyze_engagement_patterns(rules),
        'common_combinations': analyze_feature_combinations(frequent_itemsets)
    }
    
    return insights

insights = generate_musical_insights(binary_df)

# Now let's examine each type of insight separately
# 1. Musical Patterns
musical_patterns = insights['musical_patterns']
print("Top Musical Feature Relationships:")
print(musical_patterns.head().to_string())
print("\n")

# 2. Engagement Patterns
engagement_patterns = insights['engagement_patterns']
print("Top Engagement Patterns:")
print(engagement_patterns.head().to_string())
print("\n")

# 3. Common Feature Combinations
common_combinations = insights['common_combinations']
print("Most Common Feature Combinations:")
print(common_combinations.head().to_string())

# If you want to adjust the thresholds, you can run individual analyses:
# Generate frequent itemsets with different support threshold
custom_frequent_itemsets = apriori(binary_df, min_support=0.1, use_colnames=True)

# Generate rules with different confidence threshold
custom_rules = association_rules(custom_frequent_itemsets,num_itemsets=binary_df.shape[0], metric='confidence', min_threshold=0.4)

# Analyze with different lift threshold
custom_musical_patterns = analyze_musical_characteristics(custom_rules, min_lift=2.0)
custom_engagement_patterns = analyze_engagement_patterns(custom_rules, min_lift=2.0)

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def analyze_support_impact(binary_df, support_range=[0.05, 0.1, 0.15, 0.2]):
    """Analyze impact of different support thresholds"""
    support_analysis = {}
    
    for sup in support_range:
        frequent_itemsets = apriori(binary_df, min_support=sup, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", 
                                num_itemsets=binary_df.shape[0], min_threshold=0.3)
        
        support_analysis[sup] = {
            'num_itemsets': len(frequent_itemsets),
            'num_rules': len(rules),
            'avg_lift': rules['lift'].mean() if len(rules) > 0 else 0,
            'max_lift': rules['lift'].max() if len(rules) > 0 else 0
        }
    
    return pd.DataFrame(support_analysis).T

def analyze_itemset_sizes(frequent_itemsets):
    """Analyze the distribution of itemset sizes"""
    itemset_sizes = frequent_itemsets['itemsets'].apply(len)
    size_dist = itemset_sizes.value_counts().sort_index()
    
    # Add percentage calculation
    size_dist_pct = (size_dist / len(frequent_itemsets) * 100).round(2)
    return pd.DataFrame({
        'count': size_dist,
        'percentage': size_dist_pct
    })

def get_feature_category(feature):
    """Determine category for a given feature"""
    if any(x in feature for x in ['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Valence']):
        return 'audio_features'
    elif any(x in feature for x in ['Views', 'Likes', 'Comments']):
        return 'engagement'
    elif 'Stream' in feature:
        return 'performance'
    elif any(x in feature for x in ['Key', 'Duration', 'Instrumentalness', 'Speechiness', 'Liveness']):
        return 'technical'
    return 'other'

def analyze_category_relationships(rules):
    """Analyze relationships between different feature categories"""
    categories = {
        'audio_features': ['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Valence'],
        'engagement': ['Views', 'Likes', 'Comments'],
        'performance': ['Stream'],
        'technical': ['Key', 'Duration_ms', 'Instrumentalness', 'Speechiness', 'Liveness']
    }
    
    def get_category(item_set):
        # Convert frozenset to string to handle single items
        item = next(iter(item_set))
        base_feature = item.split('_')[0]  # Get base feature name without the _0, _1, etc.
        
        for cat, features in categories.items():
            if base_feature in features:
                return cat
        return 'other'
    
    # Add category information to rules
    rules['antecedent_category'] = rules['antecedents'].apply(get_category)
    rules['consequent_category'] = rules['consequents'].apply(get_category)
    
    category_metrics = rules.groupby(['antecedent_category', 'consequent_category']).agg({
        'lift': ['mean', 'max'],
        'confidence': 'mean',
        'support': 'mean'
    }).round(3)
    
    return category_metrics

def plot_market_basket_insights(rules, frequent_itemsets):
    """Create visualizations for market basket analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. Support vs Confidence scatter with lift coloring
    scatter = ax1.scatter(rules['support'], rules['confidence'], 
                         c=rules['lift'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ax=ax1, label='Lift')
    ax1.set_xlabel('Support')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Support vs Confidence (colored by Lift)')
    
    # 2. Distribution of lift values
    ax2.hist(rules['lift'], bins=30, edgecolor='black')
    mean_lift = rules['lift'].mean()
    ax2.axvline(mean_lift, color='r', linestyle='--', 
                label=f'Mean Lift: {mean_lift:.2f}')
    ax2.set_xlabel('Lift')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Lift Values')
    ax2.legend()
    
    # 3. Itemset sizes
    itemset_sizes = frequent_itemsets['itemsets'].apply(len)
    size_dist = itemset_sizes.value_counts().sort_index()
    ax3.bar(size_dist.index, size_dist.values)
    for i, v in enumerate(size_dist.values):
        ax3.text(i, v + 0.5, f'{(v/len(frequent_itemsets)*100):.1f}%', 
                ha='center')
    ax3.set_xlabel('Itemset Size')
    ax3.set_ylabel('Count')
    ax3.set_title('Frequent Itemset Sizes')
    
    # 4. Top rules by lift
    top_rules = rules.nlargest(10, 'lift')
    
    def format_rule(antecedents, consequents):
        ant = next(iter(antecedents))
        cons = next(iter(consequents))
        return f"{ant} → {cons}"
    
    rule_labels = [format_rule(row['antecedents'], row['consequents']) 
                  for _, row in top_rules.iterrows()]
    
    ax4.barh(range(len(top_rules)), top_rules['lift'])
    ax4.set_yticks(range(len(top_rules)))
    ax4.set_yticklabels(rule_labels, fontsize=8)
    ax4.set_xlabel('Lift')
    ax4.set_title('Top 10 Rules by Lift')
    
    plt.tight_layout()
    return fig

def create_rules_network(rules, min_lift=2):
    """Create network visualization of strong rules"""
    G = nx.DiGraph()
    
    # Filter significant rules
    significant_rules = rules[rules['lift'] >= min_lift]
    
    # Add edges
    for _, rule in significant_rules.iterrows():
        ant = next(iter(rule['antecedents']))
        cons = next(iter(rule['consequents']))
        G.add_edge(ant, cons, 
                  weight=rule['lift'],
                  confidence=rule['confidence'])
    
    return G

def comprehensive_market_basket_analysis(binary_df, min_support=0.05, min_confidence=0.3):
    """Main analysis function"""
    # Generate frequent itemsets and rules
    frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets,num_itemsets=binary_df.shape[0], metric="confidence", min_threshold=min_confidence)
    
    # Run analyses
    support_impact = analyze_support_impact(binary_df)
    category_relationships = analyze_category_relationships(rules)
    
    # Create visualizations
    plots = plot_market_basket_insights(rules, frequent_itemsets)
    network = create_rules_network(rules)
    
    # Generate summary
    summary = {
        'total_rules': len(rules),
        'avg_lift': rules['lift'].mean(),
        'max_lift': rules['lift'].max(),
        'avg_confidence': rules['confidence'].mean(),
        'num_itemsets': len(frequent_itemsets)
    }
    
    return {
        'summary': summary,
        'support_impact': support_impact,
        'category_relationships': category_relationships,
        'rules': rules,
        'frequent_itemsets': frequent_itemsets,
        'plots': plots,
        'network': network
    }

# Run the analysis
results = comprehensive_market_basket_analysis(binary_df)

# Print summary statistics
print("Analysis Summary:")
for key, value in results['summary'].items():
    print(f"{key}: {value:.2f}")

# Show visualizations

plt.show()


def analyze_feature_level_transitions(rules, min_lift=2.0):
    # Group features by their base characteristic
    feature_groups = {
        'energy': ['Energy_0', 'Energy_1', 'Energy_2', 'Energy_3'],
        'danceability': ['Danceability_0', 'Danceability_1', 'Danceability_2', 'Danceability_3'],
        'acousticness': ['Acousticness_0', 'Acousticness_1', 'Acousticness_2', 'Acousticness_3'],
        'valence': ['Valence_0', 'Valence_1', 'Valence_2', 'Valence_3']
    }
    
    cross_feature_rules = rules[
        rules['antecedents'].apply(lambda x: any(f in str(x) for group in feature_groups.values() for f in group)) &
        rules['consequents'].apply(lambda x: any(f in str(x) for group in feature_groups.values() for f in group)) &
        (rules['lift'] >= min_lift)
    ]
    
    return cross_feature_rules.sort_values('lift', ascending=False)

def analyze_high_performance_combinations(rules, min_confidence=0.7):
    high_performance_metrics = ['Views_3', 'Likes_3', 'Comments_3', 'Stream_3']
    
    success_rules = rules[
        rules['consequents'].apply(lambda x: any(metric in str(x) for metric in high_performance_metrics)) &
        (rules['confidence'] >= min_confidence)
    ]
    
    return success_rules.sort_values('confidence', ascending=False)

def analyze_feature_clusters(binary_df):

    feature_levels = ['_0', '_1', '_2', '_3']
    clusters = {}
    
    for level in feature_levels:
        level_features = [col for col in binary_df.columns if col.endswith(level)]
        level_data = binary_df[level_features]
        
        # Find frequent combinations within this level
        frequent_itemsets = apriori(level_data, min_support=0.1, use_colnames=True)
        clusters[f'level{level}'] = analyze_feature_combinations(frequent_itemsets)
    
    return clusters

def analyze_engagement_progression(rules):

    engagement_metrics = ['Views', 'Likes', 'Comments']
    progression_rules = {}
    
    for metric in engagement_metrics:
        # Look for rules where lower levels lead to higher levels
        metric_rules = rules[
            rules['antecedents'].apply(lambda x: f'{metric}_2' in str(x)) &
            rules['consequents'].apply(lambda x: f'{metric}_3' in str(x))
        ]
        progression_rules[metric] = metric_rules.sort_values('confidence', ascending=False)
    
    return progression_rules

def generate_comprehensive_insights(binary_df):

    # Generate base itemsets and rules
    frequent_itemsets = apriori(binary_df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, num_itemsets=binary_df.shape[0],metric='confidence', min_threshold=0.3)
    
    insights = {
        'feature_transitions': analyze_feature_level_transitions(rules),
        'high_performance': analyze_high_performance_combinations(rules),
        'feature_clusters': analyze_feature_clusters(binary_df),
        'engagement_progression': analyze_engagement_progression(rules),
        'musical_patterns': analyze_musical_characteristics(rules),
        'engagement_patterns': analyze_engagement_patterns(rules),
        'common_combinations': analyze_feature_combinations(frequent_itemsets)
    }
    
    return insights

# Function to create a visualization of the insights
def visualize_insights(insights):
    plt.figure(figsize=(15, 10))
    
    # Get top transitions
    transitions = insights['feature_transitions'].head(10)
    
    # Create pivot table for heatmap
    transition_matrix = transitions.pivot_table(
        values='lift', 
        index='antecedents', 
        columns='consequents'
    )
    
    # Create single heatmap
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0
    )
    plt.title('Feature Level Transitions')
    plt.tight_layout()
    
    return plt

print("\nPatterns Leading to High Engagement:")
print(insights['high_performance'].head().to_string())

print(df.iloc[1296])


def find_matching_songs(original_df, binary_df, feature_combination):
    """
    Find songs that match specific feature combinations from the binary encoded data
    """
    # Make sure indices match
    binary_df = binary_df.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)
    
    # Create mask for each feature level
    masks = []
    for feature, level in feature_combination.items():
        column_name = f"{feature}_{level}"
        if column_name in binary_df.columns:
            masks.append(binary_df[column_name] == 1)
    
    # Combine all masks
    if masks:
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask & mask
        
        # Get matching songs with relevant details
        matching_songs = original_df[final_mask][['Track', 'Artist', 'Album', 'Views', 'Likes', 
                                                'Energy', 'Duration_ms', 'Loudness', 'Acousticness']]
        
        return matching_songs
    return pd.DataFrame()

# Example: Find songs with the strongest pattern
# (Views_3, Energy_0) → (Duration_ms_3, Loudness_0, Acousticness_3)
feature_combination = {
    'Views': 3,
    'Energy': 0,
    'Duration_ms': 3,
    'Loudness': 0,
    'Acousticness': 3
}

matching_songs = find_matching_songs(df, binary_df, feature_combination)

if not matching_songs.empty:
    print(f"\nFound {len(matching_songs)} matching songs:")
    print("\nSample of matching songs:")
    print(matching_songs.head().to_string())
    
    # Get summary statistics of the matching songs
    print("\nSummary statistics of matching songs:")
    print(matching_songs.describe().round(2))
else:
    print("No songs found matching these criteria")