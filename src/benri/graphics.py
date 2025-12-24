import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .data import split_df

def plot_boxplots(df_list, labels, value_column='test_auc', separation=None, split = None,
                  horizontals=[], trace_line=False, title = "Boxplot comparison of different experiments", X_axis=None, Y_axis=None, 
                  TEXT_COLOR='white', BOX_COLOR='#E0E0E0', BACKGROUND_COLOR = "#1F1F1F"
                  ):
    """
    Plots boxplots of `value_column` across DataFrames (one box per DataFrame),
    grouped by `separation` (hue). If trace_line=True it traces a line through
    the medians for each separation category — aligned to the categorical x positions.

    New args:
    - X_axis: if not None, sets the x-axis label to this value
    - Y_axis: if not None, sets the y-axis label to this value
    """
    if split != None:
        # This recursive part assumes a function df_list_f exists in your scope
        try:
            for i, df in enumerate(df_list):
                df_list2, labels2 = split_df(df=df, split_by = split)
                plot_boxplots(df_list = df_list2, labels = labels2, value_column=value_column, separation=separation, split = None,
                      horizontals=horizontals, trace_line=trace_line, title = title + "  " + labels[i] , X_axis=X_axis, Y_axis=Y_axis,
                      TEXT_COLOR=TEXT_COLOR, BOX_COLOR=BOX_COLOR, BACKGROUND_COLOR=BACKGROUND_COLOR)
        except NameError as e:
            print("Error: The 'split' feature requires a function named 'df_list_f' to be defined.")
            print(e)
            return
    else:
        
        print(f'Background_color: {BACKGROUND_COLOR}')
        # Define the style dictionary for the dark background and white text
        style_dict = {
            "axes.facecolor": BACKGROUND_COLOR,    # Dark background
            "figure.facecolor":BACKGROUND_COLOR,   # Dark background
            "text.color": TEXT_COLOR,         # Default text
            "axes.labelcolor": TEXT_COLOR,    # Axis labels
            "axes.titlecolor": TEXT_COLOR,    # Title
            "xtick.color": TEXT_COLOR,        # X-axis tick labels
            "ytick.color": TEXT_COLOR,        # Y-axis tick labels
            "grid.color": TEXT_COLOR,          # Lighter grid for contrast
            "axes.edgecolor": TEXT_COLOR      # Plot border/spines
        }
        
        # --- Define props for ALL boxplot elements ---
        # These will be passed to sns.boxplot to make all lines white
        plot_props = {
            "boxprops": dict(edgecolor=BOX_COLOR),
            "whiskerprops": dict(color=BOX_COLOR),
            "capprops": dict(color=BOX_COLOR),
            "medianprops": dict(color=BOX_COLOR, linewidth=1), # Make median slightly thicker
            "flierprops": dict(markerfacecolor=BOX_COLOR, 
                               markeredgecolor=BOX_COLOR, 
                               marker='.') # Use a small dot for outliers
        }

        # Use 'with' to apply the style temporarily
        with sns.axes_style("darkgrid", style_dict):

            combined = []
            medians = []

            # --- 1. Data Preparation ---
            for i, df in enumerate(df_list):
                temp = df.copy()
                temp['DataFrame'] = labels[i]
                combined.append(temp)

                if separation is None:
                    medians.append((labels[i], temp[value_column].median()))
                else:
                    for sep_val in temp[separation].unique():
                        m = temp.loc[temp[separation] == sep_val, value_column].median()
                        medians.append((labels[i], sep_val, m))

            if separation is None:
                medians_df = pd.DataFrame(medians, columns=['DataFrame', 'Median'])
            else:
                medians_df = pd.DataFrame(medians, columns=['DataFrame', separation, 'Median'])

            all_data = pd.concat(combined, ignore_index=True)

            plt.figure(figsize=(12, 6))

            # --- 2. Plotting the Boxplot ---
            if separation is None:
                ax = sns.boxplot(
                    data=all_data,
                    x='DataFrame',
                    y=value_column,
                    hue = 'DataFrame',
                    legend = False,
                    palette='Set2',
                    order=labels,
                    **plot_props # Unpack all the white-line props
                )
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            else:
                ax = sns.boxplot(
                    data=all_data,
                    x='DataFrame',
                    y=value_column,
                    hue=separation,
                    palette='Set2',
                    order=labels,
                    **plot_props # Unpack all the white-line props
                )

                # --- 3. Plotting the Trace Line (if requested) ---
                if trace_line:
                    x = np.arange(len(labels))
                    
                    hue_levels = sorted(all_data[separation].unique())
                    palette = sns.color_palette("Set2", n_colors=len(hue_levels))
                    color_map = dict(zip(hue_levels, palette))

                    for i, sep_val in enumerate(hue_levels):
                        medians_for_hue = medians_df[medians_df[separation] == sep_val]
                        
                        ordered_medians = pd.DataFrame({'DataFrame': labels})
                        ordered_medians = ordered_medians.merge(
                            medians_for_hue, 
                            on='DataFrame', 
                            how='left'
                        )
                        y = ordered_medians['Median'].values 

                        ax.plot(
                            x, y, 
                            marker='.',        
                            linestyle='--',    
                            color=color_map[sep_val], 
                            zorder=10,         
                            alpha=0.9
                        )

            # --- 4. Plot Finalization and Styling (All Text White) ---
            
            ax.set_title(title, color=TEXT_COLOR, fontsize=16, pad=20)

            x_label = X_axis if X_axis is not None else 'DataFrame'
            y_label = Y_axis if Y_axis is not None else value_column
            ax.set_xlabel(x_label, color=TEXT_COLOR, fontsize=12, labelpad=15)
            ax.set_ylabel(y_label, color=TEXT_COLOR, fontsize=12, labelpad=15)

            ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=10)
            ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=10)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(TEXT_COLOR)

            for h_val in horizontals:
                ax.axhline(y=h_val, color=TEXT_COLOR, linestyle=':', linewidth=1, alpha=0.8)

            if separation is not None:
                legend = ax.get_legend()
                if legend:
                    title_obj = legend.get_title()
                    if title_obj:
                        title_obj.set_color(TEXT_COLOR)
                    
                    for text in legend.get_texts():
                        text.set_color(TEXT_COLOR)
                    
                    frame = legend.get_frame()
                    frame.set_facecolor(BACKGROUND_COLOR) 
                    frame.set_edgecolor(TEXT_COLOR)

        # --- 5. Show the Plot ---
        plt.tight_layout() 
        plt.show()
