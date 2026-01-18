# --- File: scripts/evaluate_config.py ---
import asyncio
from measure_variance import measure_config_variance

# --- Main execution block ---
async def main():
    """
    A wrapper script to evaluate our champion configurations from the Taguchi experiment.
    """
    # Define our two champion configurations
    zero_variance_champion = {
        "model": "gpt-5.2",
        "temperature": 0.1,
        "prompt_style": "rubric"
    }
    
    sn_ratio_champion = {
        "model": "gpt-5.2",
        "temperature": 0.7,
        "prompt_style": "simple"
    }

    # Run the evaluation for the first champion
    await measure_config_variance(zero_variance_champion, "Zero-Variance Champion")
    
    print("\n\n" + "="*50 + "\n\n") # Separator
    
    # Run the evaluation for the second champion
    await measure_config_variance(sn_ratio_champion, "S/N Ratio Champion")
    
    print(f"\n\nBaseline Variance to Beat: 0.2122")


if __name__ == "__main__":
    asyncio.run(main())