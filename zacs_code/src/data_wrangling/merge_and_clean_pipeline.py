"""
GDP and Research Papers Analysis Module

This module provides functionality to load, process, and merge GDP data with research paper data.
It handles quarterly GDP data and research paper publications, creating aligned datasets for analysis.
"""

import pandas as pd
import json
from typing import Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors"""

    pass


class GDPDataProcessor:
    """Handles loading and processing of GDP data"""

    @staticmethod
    def load_gdp_data(file_path: str, sheet_idx: int) -> pd.DataFrame:
        """
        Load GDP data from Excel file for a specific sheet and convert to monthly data

        Args:
            file_path: Path to the Excel file
            sheet_idx: 0-based index of the sheet to load

        Returns:
            DataFrame with monthly GDP data
        """
        try:
            gdp_data = pd.read_excel(
                file_path, sheet_name=sheet_idx, parse_dates=["date"]
            )

            # Standardize column names
            if "date" not in gdp_data.columns:
                gdp_data.columns = ["date", "y"]
                gdp_data["date"] = pd.to_datetime(gdp_data["date"])

            # Remove missing GDP values
            gdp_data = gdp_data.dropna(subset=["y"])

            return GDPDataProcessor._convert_to_monthly(gdp_data)

        except Exception as e:
            raise DataLoadError(f"Failed to load GDP data: {str(e)}")

    @staticmethod
    def _convert_to_monthly(quarterly_data: pd.DataFrame) -> pd.DataFrame:
        """Convert quarterly GDP data to monthly by forward-filling values"""
        monthly_gdp = []

        for _, quarter_value in quarterly_data.iterrows():
            for month in range(3):
                monthly_gdp.append(
                    {
                        "date": quarter_value["date"] + pd.DateOffset(months=month),
                        "gdp": quarter_value["y"],
                    }
                )

        return pd.DataFrame(monthly_gdp)


class PapersDataProcessor:
    """Handles loading and processing of research papers data"""

    @staticmethod
    def load_papers_data(file_path: str) -> pd.DataFrame:
        """
        Load papers from a JSON file

        Args:
            file_path: Path to the JSON file

        Returns:
            DataFrame containing paper data
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                papers = json.load(file)

            papers_df = pd.DataFrame(papers)
            papers_df["publication_date"] = pd.to_datetime(
                papers_df["publication_date"], errors="coerce"
            )

            # Drop rows where date conversion failed
            papers_df = papers_df.dropna(subset=["publication_date"])
            return papers_df

        except Exception as e:
            raise DataLoadError(f"Failed to load papers data: {str(e)}")

    @staticmethod
    def prepare_paper_data(papers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare paper data for analysis

        Args:
            papers_df: DataFrame containing paper data

        Returns:
            Processed DataFrame with standardized date column
        """
        papers_temp = papers_df.copy()

        # Find and standardize date column
        date_cols = [col for col in papers_temp.columns if "date" in col.lower()]
        if not date_cols:
            raise ValueError("No date column found in papers DataFrame")

        date_col = date_cols[0]
        logger.info(f"Using {date_col} as date column")

        papers_temp[date_col] = pd.to_datetime(papers_temp[date_col])
        papers_temp = papers_temp.rename(columns={date_col: "date"})

        logger.info(f"Processed {len(papers_temp)} individual papers")
        logger.info(f"Columns in papers_temp: {papers_temp.columns.tolist()}")

        return papers_temp


class DataMerger:
    """Handles merging of GDP and papers data"""

    @staticmethod
    def merge_with_gdp(
        papers_df: pd.DataFrame, gdp_train_df: pd.DataFrame, gdp_test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge paper records with GDP data and split into train/test sets

        Args:
            papers_df: DataFrame containing paper data
            gdp_train_df: Training GDP data
            gdp_test_df: Test GDP data

        Returns:
            Tuple of (training DataFrame, test DataFrame)
        """
        papers = papers_df.copy()
        gdp_train = gdp_train_df.copy()
        gdp_test = gdp_test_df.copy()

        # Ensure dates are datetime
        for df in [papers, gdp_train, gdp_test]:
            df["date"] = pd.to_datetime(df["date"])

        DataMerger._log_date_ranges(papers, gdp_train, gdp_test)

        # Process GDP data
        gdp_train_processed = DataMerger._process_gdp(gdp_train)
        gdp_test_processed = DataMerger._process_gdp(gdp_test)

        # Prepare papers for merging
        papers["month_start"] = papers["date"].dt.to_period("M").dt.to_timestamp()

        # Split and merge data
        train_cutoff = gdp_train["date"].max()
        test_start = gdp_test["date"].min()

        papers_train = papers[papers["date"] <= train_cutoff].copy()
        papers_test = papers[papers["date"] >= test_start].copy()

        train_final = DataMerger._merge_subset(papers_train, gdp_train_processed)
        test_final = DataMerger._merge_subset(papers_test, gdp_test_processed)

        DataMerger._validate_merged_data(train_final, test_final)

        return train_final, test_final

    @staticmethod
    def _process_gdp(gdp_df: pd.DataFrame) -> pd.DataFrame:
        """Process GDP data with linear interpolation"""
        date_range = pd.date_range(
            start=gdp_df["date"].min(), end=gdp_df["date"].max(), freq="MS"
        )

        gdp = gdp_df.set_index("date")
        gdp_full = pd.DataFrame(index=date_range).join(gdp)
        gdp_processed = gdp_full.interpolate(method="linear")
        gdp_processed.columns = ["y"]

        return gdp_processed.reset_index().rename(columns={"index": "date"})

    @staticmethod
    def _merge_subset(papers_df: pd.DataFrame, gdp_df: pd.DataFrame) -> pd.DataFrame:
        """Merge papers with GDP data"""
        merged = papers_df.merge(
            gdp_df, left_on="month_start", right_on="date", how="left"
        )

        merged.drop(["date_y", "month_start"], axis=1, inplace=True)
        merged.rename(columns={"date_x": "date"}, inplace=True)

        return merged

    @staticmethod
    def _log_date_ranges(
        papers_df: pd.DataFrame, gdp_train: pd.DataFrame, gdp_test: pd.DataFrame
    ):
        """Log date ranges for all datasets"""
        logger.info("\nDate ranges:")
        logger.info(f"Papers: {papers_df['date'].min()} to {papers_df['date'].max()}")
        logger.info(
            f"GDP train: {gdp_train['date'].min()} to {gdp_train['date'].max()}"
        )
        logger.info(f"GDP test: {gdp_test['date'].min()} to {gdp_test['date'].max()}")

    @staticmethod
    def _validate_merged_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate merged datasets and log results"""
        logger.info("\nTrain/Test Split Results:")
        logger.info(f"Training set shape: {train_df.shape}")
        logger.info(
            f"Training set date range: {train_df['date'].min()} to {train_df['date'].max()}"
        )
        logger.info(f"Training papers with GDP data: {train_df.dropna().shape[0]}")

        logger.info(f"\nTest set shape: {test_df.shape}")
        logger.info(
            f"Test set date range: {test_df['date'].min()} to {test_df['date'].max()}"
        )
        logger.info(f"Test papers with GDP data: {test_df.dropna().shape[0]}")

        # Check for missing GDP data
        for name, df in [("training", train_df), ("test", test_df)]:
            missing = df[df["y"].isna()]
            if len(missing) > 0:
                logger.warning(
                    f"\nWarning: {len(missing)} {name} papers missing GDP data"
                )
                logger.warning(
                    f"Missing {name} dates: {missing['date'].min()} to {missing['date'].max()}"
                )


def main():
    """Main execution function"""
    try:
        # Define file paths
        data_dir = Path("data")
        gdp_file = data_dir / "gdp_base.xlsx"
        papers_file = data_dir / "economics_abstracts_historical.json"

        # Load data
        gdp_processor = GDPDataProcessor()
        papers_processor = PapersDataProcessor()

        gdp_train = gdp_processor.load_gdp_data(gdp_file, sheet_idx=2)
        gdp_test = gdp_processor.load_gdp_data(gdp_file, sheet_idx=3)
        papers_df = papers_processor.load_papers_data(papers_file)

        # Process and merge data
        papers = papers_processor.prepare_paper_data(papers_df)
        train_merged, test_merged = DataMerger.merge_with_gdp(
            papers, gdp_train, gdp_test
        )

        # Save results
        train_merged.to_csv(data_dir / "train_merged.csv", index=False)
        test_merged.to_csv(data_dir / "test_merged.csv", index=False)

        logger.info("Data processing completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":  # Fixed from "main"
    main()
