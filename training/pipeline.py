import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_classifier import main as train_main
from finetune_classifier import run_fine_tuning
from hyperparameter_tuning import run_hyperparameter_tuning
import yaml
import json
from datetime import datetime
import os

def run_full_pipeline():
    
    # STEP 0: Freezing
    print("# STEP 0: Baseline Training (no_freeze)")

    baseline_metrics = run_baseline_training()
    
    print(f"\n✓ Baseline completed.\nResults:{baseline_metrics}")

    # STEP 1: Fine-Tuning 
    print("# STEP 1: Fine-Tuning")
    
    finetune_results = run_fine_tuning()
    
    print(f"\n✓ Fine-tuning completed.\nResults:{finetune_results}")
    
    # STEP 2: Fine-Tuning and Hyperparameter Tuning 
    print("# STEP 2: Hyperparameter Tuning with Best Strategy")
    
    best_config, tuning_results = run_hyperparameter_tuning()
    
    print(f"✓ Hyperparameter tuning completed.\nResults:{tuning_results}\n")
    print(f'BEST CONFIGURATION:/n{best_config}')
    
    # STEP 3: Final Report
    print("# STEP 3: Final Report")
    
    generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config)
    
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"End time: {datetime.now()}")

def run_baseline_training():

    with open('experiments/baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    
    metrics = train_main(config)
    
    return {
            'accuracy': metrics['accuracy'],
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'val_loss': metrics['val_loss']
            }


FINAL_METRICS = "Final Metrics:\n"
def generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config):
    # --- Generate improved image report with graphs and schemas ---
    report_dir = "results/baseline/final_report"
    os.makedirs(report_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        # Prepare metrics for each step
        steps = ['Baseline', 'Fine-tuned', 'Hyperparam Tuned']
        metrics = ['accuracy', 'f1', 'roc_auc', 'val_loss']
        values = [
            [baseline_metrics['accuracy'], baseline_metrics['f1'], baseline_metrics['roc_auc'], baseline_metrics['val_loss']],
            [finetune_results['accuracy'], finetune_results['f1'], finetune_results['roc_auc'], finetune_results['val_loss']],
            [tuning_results['accuracy'], tuning_results['f1'], tuning_results['roc_auc'], tuning_results['val_loss']]
        ]
        df = pd.DataFrame(values, columns=metrics, index=steps)

        # Set up figure
        fig = plt.figure(constrained_layout=True, figsize=(14, 10))
        gs = fig.add_gridspec(3, 2)

        # Title
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.set_title('FINAL OPTIMIZATION REPORT', fontsize=22, fontweight='bold', pad=20)

        # Bar chart for metrics
        ax1 = fig.add_subplot(gs[1, 0])
        df_plot = df.drop('val_loss', axis=1)
        df_plot.plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Metrics')
        ax1.legend(loc='lower right')
        ax1.set_xticklabels(df_plot.index, rotation=0)

        # Validation loss comparison
        ax2 = fig.add_subplot(gs[1, 1])
        sns.barplot(x=df.index, y=df['val_loss'], ax=ax2, palette='viridis')
        ax2.set_title('Validation Loss Comparison')
        ax2.set_ylabel('Val Loss')
        for i, v in enumerate(df['val_loss']):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

        # Table for best hyperparameters
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.axis('off')
        hp_table = [
            ['Learning Rate', best_config['learning_rate']],
            ['Batch Size', best_config['batch_size']],
            ['Weight Decay', best_config['weight_decay']],
            ['Optimizer', best_config['optimizer']]
        ]
        table = ax3.table(cellText=hp_table, colLabels=['Hyperparameter', 'Value'], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        ax3.set_title('Best Hyperparameter Configuration', fontsize=14, pad=10)

        # Improvement summary
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        improvement_finetune = (finetune_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        improvement_final = (tuning_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        summary_text = (
            f"F1 Score Improvement:\n"
            f"- Baseline: {baseline_metrics['f1']:.4f}\n"
            f"- Fine-tuned: {finetune_results['f1']:.4f} ({improvement_finetune:+.2f}%)\n"
            f"- Hyperparam Tuned: {tuning_results['f1']:.4f} ({improvement_final:+.2f}%)\n\n"
            f"Model location:\n{model_loc}"
        )
        ax4.text(0, 1, summary_text, fontsize=12, va='top', ha='left', wrap=True)

        # Save image
        image_path = os.path.join(report_dir, "optimization_report.png")
        plt.savefig(image_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"✓ Improved report image saved: {image_path}")
    except Exception as e:
        print(f"[WARN] Could not generate improved report image: {e}")
    
    
    
    report_path = os.path.join(report_dir, "optimization_report.txt")
    

    # Write text report
    with open(report_path, 'w') as f:
        f.write("FINAL OPTIMIZATION REPORT\n")
        f.write("STEP 0: BASELINE TRAINING (freeze)\n")
        f.write(f"  - Accuracy: {baseline_metrics['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {baseline_metrics['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {baseline_metrics['roc_auc']:.4f}\n")
        f.write(f"  - Val Loss: {baseline_metrics['val_loss']:.4f}\n\n")
        f.write("STEP 1: FINE-TUNING\n")
        f.write(FINAL_METRICS)
        f.write(f"  - Accuracy: {finetune_results['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {finetune_results['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {finetune_results['roc_auc']:.4f}\n")
        f.write(f"  - Val Loss: {finetune_results['val_loss']:.4f}\n\n")
        f.write("STEP 2: HYPERPARAMETER TUNING\n")
        f.write(f"Best hyperparameter configuration:\n{best_config}\n")
        f.write(FINAL_METRICS)
        f.write(f"  - Accuracy: {tuning_results['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {tuning_results['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {tuning_results['roc_auc']:.4f}\n")
        f.write(f"  - Validation Loss: {tuning_results['val_loss']:.4f}\n\n")
        f.write("COMPARISON: Baseline vs Fine-tuned vs Fine Tuned and Hyperparameter Tuned\n")
        improvement_finetune = (finetune_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        improvement_final = (tuning_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        f.write(f"  Baseline F1: {baseline_metrics['f1']:.4f}\n")
        f.write(f" Fine Tune F1: {finetune_results['f1']:.4f} ({improvement_finetune:+.2f}%)\n")
        f.write(f"  After Hyperparameter Tuning F1: {tuning_results['f1']:.4f} ({improvement_final:+.2f}%)\n\n")
        model_loc = "results/hyperparameter_tuning/{}/classifier.pth".format(
            f"lr_{best_config['learning_rate']}_bs_{best_config['batch_size']}_wd_{best_config['weight_decay']}_{best_config['optimizer']}_cwr".replace('.', '_')
        )
        f.write(f"Model location: {model_loc}\n")
        f.write("Best hyperparameter configuration:\n")
        f.write(f"  - Learning Rate: {best_config['learning_rate']}\n")
        f.write(f"  - Batch Size: {best_config['batch_size']}\n")
        f.write(f"  - Weight Decay: {best_config['weight_decay']}\n")
        f.write(f"  - Momentum: {best_config['momentum']}\n")
        f.write(f"  - Optimizer: {best_config['optimizer']}\n")
        f.write(FINAL_METRICS)
        f.write(f"  - Accuracy: {tuning_results['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {tuning_results['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {tuning_results['roc_auc']:.4f}\n")
        f.write(f"  - Validation Loss: {tuning_results['val_loss']:.4f}\n\n")

    print(f"✓ Report saved: {report_path}")

    # Save summary as JSON
    summary_path = os.path.join(report_dir, "best_config.json")
    summary = {
        'best_hyperparameters': {
            'learning_rate': float(best_config['learning_rate']),
            'batch_size': int(best_config['batch_size']),
            'weight_decay': float(best_config['weight_decay']),
            'momentum': float(best_config['momentum']),
            'optimizer': float(best_config['optimizer']),
        },
        'final_metrics': {
            'accuracy': float(best_config['accuracy']),
            'precision': float(best_config['precision']),
            'recall': float(best_config['recall']),
            'f1_score': float(best_config['f1']),
            'roc_auc': float(best_config['roc_auc']),
            'val_loss': float(best_config['val_loss'])
        },
        'model_path': model_loc
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Config saved: {summary_path}")

if __name__ == '__main__':
    run_full_pipeline()
