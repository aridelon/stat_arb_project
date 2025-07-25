import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.sector_screener import SectorScreener

from config import financials, energy, technology, consumer_staples, healthcare
from config import reits, industrials, utilities, materials

SECTORS = {
    'financials': (financials.SECTOR_NAME, financials.FOLDER_NAME, financials.TICKERS),
    'energy': (energy.SECTOR_NAME, energy.FOLDER_NAME, energy.TICKERS),
    'technology': (technology.SECTOR_NAME, technology.FOLDER_NAME, technology.TICKERS),
    'consumer_staples': (consumer_staples.SECTOR_NAME, consumer_staples.FOLDER_NAME, consumer_staples.TICKERS),
    'healthcare': (healthcare.SECTOR_NAME, healthcare.FOLDER_NAME, healthcare.TICKERS),
    'reits': (reits.SECTOR_NAME, reits.FOLDER_NAME, reits.TICKERS),
    'industrials': (industrials.SECTOR_NAME, industrials.FOLDER_NAME, industrials.TICKERS),
    'utilities': (utilities.SECTOR_NAME, utilities.FOLDER_NAME, utilities.TICKERS),
    'materials': (materials.SECTOR_NAME, materials.FOLDER_NAME, materials.TICKERS),
}

def main():
    if len(sys.argv) < 2:
        print("python3 run_sector_analysis.py <sector> [year]")
        return
    
    sector = sys.argv[1].lower()
    year = sys.argv[2] if len(sys.argv) > 2 else '2024'
    
    if sector not in SECTORS:
        print(f"Unknown sector: {sector}")
        print(f"Available sectors: {list(SECTORS.keys())}")
        return

    name, folder, tickers = SECTORS[sector]
    screener = SectorScreener(name, folder, tickers, year=year)
    screener.run_analysis()

if __name__ == "__main__":
    main()