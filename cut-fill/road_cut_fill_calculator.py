import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

class RoadCutFillAnalyzer:
    """Class to analyze cut and fill requirements for road construction"""
    
    def __init__(self, road_width=10, side_slope=1.5, shrinkage_factor=0.85, 
                 swell_factor=1.25, station_interval=20):
        """
        Initialize the cut-fill analyzer
        
        Parameters:
        - road_width: Width of road template (m)
        - side_slope: Side slope ratio (horizontal:vertical)
        - shrinkage_factor: Soil shrinkage factor for fill (0.7-0.9)
        - swell_factor: Soil swell factor for cut (1.2-1.4)
        - station_interval: Distance between stations (m)
        """
        self.road_width = road_width
        self.side_slope = side_slope
        self.shrinkage_factor = shrinkage_factor
        self.swell_factor = swell_factor
        self.station_interval = station_interval
        
    def load_data(self, existing_file, design_file):
        """
        Load existing ground and design road data from CSV files
        
        CSV format expected:
        station, elevation, northing, easting (optional)
        or
        distance, elevation, x, y (optional)
        """
        # Load existing ground data
        self.existing_df = pd.read_csv(existing_file)
        self.design_df = pd.read_csv(design_file)
        
        # Standardize column names
        self._standardize_columns()
        
        # Sort by station/distance
        self.existing_df = self.existing_df.sort_values('station')
        self.design_df = self.design_df.sort_values('station')
        
        # Interpolate design to match existing stations if needed
        self._align_stations()
        
        print(f" Loaded {len(self.existing_df)} existing ground points")
        print(f" Loaded {len(self.design_df)} design road points")
        print(f"   Station range: {self.existing_df['station'].min():.2f} - {self.existing_df['station'].max():.2f} m")
        
    def _standardize_columns(self):
        """Standardize column names to 'station' and 'elevation'"""
        # Map possible column names
        station_cols = ['station', 'distance', 'chainage', 'sta', 'km', 'm', 'dist']
        elev_cols = ['elevation', 'height', 'z', 'level', 'elev', 'altitude']
        
        # Find station column
        for df in [self.existing_df, self.design_df]:
            # Try to find station column
            found_station = False
            for col in station_cols:
                if col in df.columns:
                    df.rename(columns={col: 'station'}, inplace=True)
                    found_station = True
                    break
            
            if not found_station:
                # If no station column, use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df.rename(columns={numeric_cols[0]: 'station'}, inplace=True)
                    print(f"   Using '{numeric_cols[0]}' as station column")
            
            # Find elevation column
            found_elev = False
            for col in elev_cols:
                if col in df.columns:
                    df.rename(columns={col: 'elevation'}, inplace=True)
                    found_elev = True
                    break
            
            if not found_elev:
                # If no elevation column, use the second numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    df.rename(columns={numeric_cols[1]: 'elevation'}, inplace=True)
                    print(f"   Using '{numeric_cols[1]}' as elevation column")
            
            # Ensure numeric
            df['station'] = pd.to_numeric(df['station'], errors='coerce')
            df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
            
            # Drop any rows with NaN
            df.dropna(subset=['station', 'elevation'], inplace=True)
    
    def _align_stations(self):
        """Interpolate design elevations at existing ground stations"""
        # Create interpolation function for design data
        design_interp = interpolate.interp1d(
            self.design_df['station'], 
            self.design_df['elevation'],
            kind='linear',
            fill_value='extrapolate'
        )
        
        # Get design elevations at existing stations
        self.design_at_existing = design_interp(self.existing_df['station'])
        
        # Create combined dataframe
        self.analysis_df = pd.DataFrame({
            'station': self.existing_df['station'],
            'existing_elev': self.existing_df['elevation'],
            'design_elev': self.design_at_existing
        })
    
    def calculate_cut_fill(self):
        """Calculate cut and fill at each station"""
        # Calculate difference (positive = cut, negative = fill)
        self.analysis_df['difference'] = self.analysis_df['existing_elev'] - self.analysis_df['design_elev']
        
        # Separate cut and fill
        self.analysis_df['cut_depth'] = np.maximum(0, self.analysis_df['difference'])
        self.analysis_df['fill_depth'] = np.maximum(0, -self.analysis_df['difference'])
        
        # Calculate cross-sectional area at each station
        # Assuming trapezoidal road template
        self.analysis_df['cut_area'] = self._calculate_cross_section_area(
            self.analysis_df['cut_depth'].values
        )
        self.analysis_df['fill_area'] = self._calculate_cross_section_area(
            self.analysis_df['fill_depth'].values
        )
        
        # Calculate volume between stations (trapezoidal rule)
        self._calculate_volumes()
        
        # Calculate cumulative volumes
        self.analysis_df['cumulative_cut'] = self.analysis_df['cut_volume'].cumsum()
        self.analysis_df['cumulative_fill'] = self.analysis_df['fill_volume'].cumsum()
        self.analysis_df['net_volume'] = self.analysis_df['cumulative_cut'] - self.analysis_df['cumulative_fill']
        
        # Adjust for soil factors
        self.analysis_df['cut_bank_volume'] = self.analysis_df['cut_volume']  # Bank cubic meters
        self.analysis_df['fill_compacted_volume'] = self.analysis_df['fill_volume']  # Compacted cubic meters
        self.analysis_df['cut_loose_volume'] = self.analysis_df['cut_volume'] * self.swell_factor  # Loose cubic meters
        self.analysis_df['fill_bank_volume'] = self.analysis_df['fill_volume'] / self.shrinkage_factor  # Bank cubic meters
        
    def _calculate_cross_section_area(self, depths):
        """Calculate cross-sectional area for cut/fill at a station"""
        areas = []
        for depth in depths:
            if depth <= 0:
                areas.append(0)
            else:
                # Trapezoidal area: road_width * depth + side_slope * depth^2
                area = self.road_width * depth + self.side_slope * depth**2
                areas.append(area)
        return np.array(areas)
    
    def _calculate_volumes(self):
        """Calculate volumes between stations using average end area method"""
        cut_volumes = []
        fill_volumes = []
        
        for i in range(len(self.analysis_df) - 1):
            # Distance between stations
            dist = self.analysis_df.iloc[i+1]['station'] - self.analysis_df.iloc[i]['station']
            
            # Average end area method
            avg_cut_area = (self.analysis_df.iloc[i]['cut_area'] + self.analysis_df.iloc[i+1]['cut_area']) / 2
            avg_fill_area = (self.analysis_df.iloc[i]['fill_area'] + self.analysis_df.iloc[i+1]['fill_area']) / 2
            
            cut_volumes.append(avg_cut_area * dist)
            fill_volumes.append(avg_fill_area * dist)
        
        # Add 0 for last point (no volume past last station)
        cut_volumes.append(0)
        fill_volumes.append(0)
        
        self.analysis_df['cut_volume'] = cut_volumes
        self.analysis_df['fill_volume'] = fill_volumes
    
    def generate_report(self):
        """Generate detailed cut-fill report"""
        print("\n" + "="*80)
        print("ROAD CUT AND FILL ANALYSIS REPORT")
        print("="*80)
        
        # Summary statistics
        total_cut = self.analysis_df['cut_volume'].sum()
        total_fill = self.analysis_df['fill_volume'].sum()
        net_volume = total_cut - total_fill
        max_cut_depth = self.analysis_df['cut_depth'].max()
        max_fill_depth = self.analysis_df['fill_depth'].max()
        
        # Store for mass haul calculation
        self.total_cut = total_cut
        self.total_fill = total_fill
        
        print(f"\n SUMMARY STATISTICS:")
        print(f"   Total Length: {self.analysis_df['station'].max() - self.analysis_df['station'].min():.2f} m")
        print(f"   Number of Stations: {len(self.analysis_df)}")
        print(f"   Road Width: {self.road_width} m")
        print(f"   Side Slope: 1:{self.side_slope}")
        
        print(f"\n VOLUME SUMMARY (Bank Cubic Meters):")
        print(f"   Total Cut Volume: {total_cut:,.2f} m³")
        print(f"   Total Fill Volume: {total_fill:,.2f} m³")
        print(f"   Net Volume (Cut - Fill): {net_volume:,.2f} m³")
        
        if net_volume > 0:
            print(f"   ➤ EXCESS CUT: {net_volume:,.2f} m³ (need waste area)")
        elif net_volume < 0:
            print(f"   ➤ EXCESS FILL: {abs(net_volume):,.2f} m³ (need borrow area)")
        else:
            print(f"   ➤ BALANCED CUT AND FILL")
        
        print(f"\n ADJUSTED VOLUMES (with soil factors):")
        print(f"   Cut (Loose Volume, swell factor={self.swell_factor}): {total_cut * self.swell_factor:,.2f} m³")
        print(f"   Fill (Bank Volume, shrinkage factor={self.shrinkage_factor}): {total_fill / self.shrinkage_factor:,.2f} m³")
        
        print(f"\n MAXIMUM DEPTHS:")
        if max_cut_depth > 0:
            print(f"   Maximum Cut Depth: {max_cut_depth:.2f} m at station {self.analysis_df.loc[self.analysis_df['cut_depth'].idxmax(), 'station']:.2f}")
        if max_fill_depth > 0:
            print(f"   Maximum Fill Depth: {max_fill_depth:.2f} m at station {self.analysis_df.loc[self.analysis_df['fill_depth'].idxmax(), 'station']:.2f}")
        
        # Calculate mass haul
        self._calculate_mass_haul()
    
    def _calculate_mass_haul(self):
        """Calculate mass haul diagram statistics"""
        # Find balance points (where cumulative cut = cumulative fill)
        balance_points = []
        prev_net = 0
        
        for i, row in self.analysis_df.iterrows():
            if i > 0:
                if (prev_net <= 0 and row['net_volume'] >= 0) or (prev_net >= 0 and row['net_volume'] <= 0):
                    # Linear interpolation to find exact balance point
                    prev_row = self.analysis_df.iloc[i-1]
                    t = -prev_row['net_volume'] / (row['net_volume'] - prev_row['net_volume'])
                    balance_station = prev_row['station'] + t * (row['station'] - prev_row['station'])
                    balance_points.append(balance_station)
            prev_net = row['net_volume']
        
        print(f"\n MASS HAUL ANALYSIS:")
        if balance_points:
            print(f"   Balance Points at stations: {[f'{bp:.2f}' for bp in balance_points]}")
        else:
            print(f"   No balance points found - project is either all cut or all fill")
        
        # Calculate average haul distance
        if len(self.analysis_df) > 1 and hasattr(self, 'total_cut') and self.total_cut > 0:
            # Calculate weighted average haul distance
            total_station_meters = 0
            total_volume = 0
            
            for i in range(len(self.analysis_df) - 1):
                avg_cut = (self.analysis_df.iloc[i]['cut_volume'] + self.analysis_df.iloc[i+1]['cut_volume']) / 2
                if avg_cut > 0:
                    station_mid = (self.analysis_df.iloc[i]['station'] + self.analysis_df.iloc[i+1]['station']) / 2
                    total_station_meters += avg_cut * station_mid
                    total_volume += avg_cut
            
            if total_volume > 0:
                avg_haul = total_station_meters / total_volume
                print(f"   Average Haul Distance: {avg_haul:.2f} m")
    
    def plot_profiles(self, save_path=None):
        """Generate comprehensive cut-fill plots"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Existing vs Design Profile
        ax1 = axes[0]
        ax1.plot(self.analysis_df['station'], self.analysis_df['existing_elev'], 
                'g-', linewidth=2, label='Existing Ground', alpha=0.7)
        ax1.plot(self.analysis_df['station'], self.analysis_df['design_elev'], 
                'r-', linewidth=2, label='Design Road', alpha=0.7)
        
        # Fill between to show cut/fill areas
        ax1.fill_between(self.analysis_df['station'], 
                        self.analysis_df['existing_elev'], 
                        self.analysis_df['design_elev'],
                        where=(self.analysis_df['existing_elev'] > self.analysis_df['design_elev']),
                        color='red', alpha=0.3, label='Cut Area')
        ax1.fill_between(self.analysis_df['station'], 
                        self.analysis_df['existing_elev'], 
                        self.analysis_df['design_elev'],
                        where=(self.analysis_df['existing_elev'] < self.analysis_df['design_elev']),
                        color='blue', alpha=0.3, label='Fill Area')
        
        ax1.set_xlabel('Station (m)')
        ax1.set_ylabel('Elevation (m)')
        ax1.set_title('Existing Ground vs Design Road Profile')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cut and Fill Depths
        ax2 = axes[1]
        ax2.bar(self.analysis_df['station'], self.analysis_df['cut_depth'], 
                width=self.station_interval/2, color='red', alpha=0.7, label='Cut Depth')
        ax2.bar(self.analysis_df['station'], -self.analysis_df['fill_depth'], 
                width=self.station_interval/2, color='blue', alpha=0.7, label='Fill Depth')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Station (m)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Cut and Fill Depths')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Volumes (Mass Haul Diagram)
        ax3 = axes[2]
        ax3.plot(self.analysis_df['station'], self.analysis_df['cumulative_cut'], 
                'r-', linewidth=2, label='Cumulative Cut')
        ax3.plot(self.analysis_df['station'], self.analysis_df['cumulative_fill'], 
                'b-', linewidth=2, label='Cumulative Fill')
        ax3.plot(self.analysis_df['station'], self.analysis_df['net_volume'], 
                'g-', linewidth=2, label='Net Volume (Cut - Fill)')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Station (m)')
        ax3.set_ylabel('Volume (m³)')
        ax3.set_title('Cumulative Volumes (Mass Haul Diagram)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n Plots saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, output_file='cut_fill_results.csv'):
        """Export analysis results to CSV"""
        export_df = self.analysis_df.copy()
        export_df = export_df.round(3)
        export_df.to_csv(output_file, index=False)
        print(f"\n Results exported to: {output_file}")
        
        # Also create a summary table
        summary = {
            'Parameter': [
                'Total Length (m)',
                'Total Cut (m³)',
                'Total Fill (m³)',
                'Net Volume (m³)',
                'Max Cut Depth (m)',
                'Max Fill Depth (m)',
                'Road Width (m)',
                'Side Slope',
                'Shrinkage Factor',
                'Swell Factor'
            ],
            'Value': [
                f"{self.analysis_df['station'].max() - self.analysis_df['station'].min():.2f}",
                f"{self.analysis_df['cut_volume'].sum():,.2f}",
                f"{self.analysis_df['fill_volume'].sum():,.2f}",
                f"{self.analysis_df['cut_volume'].sum() - self.analysis_df['fill_volume'].sum():,.2f}",
                f"{self.analysis_df['cut_depth'].max():.2f}",
                f"{self.analysis_df['fill_depth'].max():.2f}",
                f"{self.road_width}",
                f"1:{self.side_slope}",
                f"{self.shrinkage_factor}",
                f"{self.swell_factor}"
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('cut_fill_summary.csv', index=False)
        print(f" Summary exported to: cut_fill_summary.csv")
        
        # Also export a simplified cut/fill table for construction
        simple_export = self.analysis_df[['station', 'cut_depth', 'fill_depth', 'cut_volume', 'fill_volume']].copy()
        simple_export.columns = ['Station (m)', 'Cut Depth (m)', 'Fill Depth (m)', 'Cut Volume (m³)', 'Fill Volume (m³)']
        simple_export.to_csv('cut_fill_construction.csv', index=False)
        print(f" Construction table exported to: cut_fill_construction.csv")

def create_sample_data():
    """Create sample existing and design data for testing"""
    
    print("\n Creating sample data files...")
    
    # Sample existing ground data
    stations = np.arange(0, 1001, 25)  # 0 to 1000m at 25m intervals
    
    # Create undulating existing ground
    existing_elev = 100 + 5 * np.sin(stations / 100) + 2 * np.sin(stations / 50)
    existing_elev += np.random.normal(0, 0.5, len(stations))
    
    existing_df = pd.DataFrame({
        'station': stations,
        'elevation': existing_elev
    })
    existing_df.to_csv('sample_existing_ground.csv', index=False)
    
    # Create design road with gentle grade and some vertical curves
    design_stations = np.arange(0, 1001, 50)
    
    # Design with sag curve and crest curve
    design_elev = []
    for s in design_stations:
        if s < 300:
            # Initial downhill grade
            elev = 98 - 0.01 * s
        elif s < 600:
            # Sag curve
            t = (s - 300) / 300
            elev = 95 + 5 * np.sin(t * np.pi / 2)
        else:
            # Uphill grade
            elev = 100 + 0.02 * (s - 600)
        design_elev.append(elev)
    
    design_df = pd.DataFrame({
        'station': design_stations,
        'elevation': design_elev
    })
    design_df.to_csv('sample_design_road.csv', index=False)
    
    print(" Sample data files created:")
    print("   - sample_existing_ground.csv (100 points, undulating terrain)")
    print("   - sample_design_road.csv (21 points, design road with vertical curves)")
    
    # Show sample data preview
    print("\n Sample existing ground data (first 5 rows):")
    print(existing_df.head())
    print("\n Sample design road data (first 5 rows):")
    print(design_df.head())
    
    return existing_df, design_df

def main():
    """Main function to run the cut-fill analysis"""
    
    print("="*80)
    print("ROAD CUT AND FILL VOLUME CALCULATOR")
    print("="*80)
    print("This program calculates cut and fill volumes for road construction")
    print("based on existing ground and design road elevations.")
    
    # Option to create sample data
    create_samples = input("\n Create sample data files for testing? (y/n): ").lower().strip()
    if create_samples == 'y':
        create_sample_data()
    
    # Get input files
    print("\n Enter input file paths:")
    print("   (CSV files should have columns: station/distance and elevation)")
    
    existing_file = input("   Existing ground CSV file: ").strip()
    design_file = input("   Design road CSV file: ").strip()
    
    # Get road parameters
    print("\n Enter road parameters (press Enter for defaults):")
    
    try:
        road_width = float(input("   Road width (m) [default=10]: ") or "10")
    except:
        road_width = 10
    
    try:
        side_slope = float(input("   Side slope (H:V) [default=1.5]: ") or "1.5")
    except:
        side_slope = 1.5
    
    try:
        shrinkage = float(input("   Shrinkage factor (0.7-0.9) [default=0.85]: ") or "0.85")
    except:
        shrinkage = 0.85
    
    try:
        swell = float(input("   Swell factor (1.2-1.4) [default=1.25]: ") or "1.25")
    except:
        swell = 1.25
    
    try:
        interval = float(input("   Station interval (m) [default=20]: ") or "20")
    except:
        interval = 20
    
    # Initialize analyzer
    analyzer = RoadCutFillAnalyzer(
        road_width=road_width,
        side_slope=side_slope,
        shrinkage_factor=shrinkage,
        swell_factor=swell,
        station_interval=interval
    )
    
    # Load data
    try:
        analyzer.load_data(existing_file, design_file)
    except Exception as e:
        print(f"\n Error loading data: {e}")
        print("   Please check that your CSV files exist and have the correct format.")
        return
    
    # Calculate cut and fill
    analyzer.calculate_cut_fill()
    
    # Generate report
    analyzer.generate_report()
    
    # Plot results
    plot_choice = input("\n Generate plots? (y/n): ").lower().strip()
    if plot_choice == 'y':
        analyzer.plot_profiles(save_path='cut_fill_analysis.png')
    
    # Export results
    export_choice = input("\n Export results to CSV? (y/n): ").lower().strip()
    if export_choice == 'y':
        analyzer.export_results()
    
    print("\n You're welcome you bot!")

if __name__ == "__main__":
    main()
