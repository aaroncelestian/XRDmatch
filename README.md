# XRD Phase Matcher

A comprehensive GUI application for X-ray diffraction phase matching using the American Mineralogist Crystal Structure Database (AMCSD).

## Features

### 🔍 **Pattern Analysis**
- **Drag & Drop Support**: Simply drag XRD files into the application
- Load diffraction patterns from various formats (XY, XYE, TXT, DAT, CSV)
- Support for error bars (XYE format) with automatic detection
- Automatic peak detection with customizable parameters
- Multiple wavelength support (Cu Kα, Co Kα, Fe Kα, Mo Kα, custom)
- Interactive pattern visualization with matplotlib

### 🔧 **Data Processing**
- **Interactive ALS Background Subtraction**: Real-time preview with adjustable parameters
- **Smoothing and Noise Reduction**: Additional signal processing options
- **Real-time Preview**: See changes instantly as you adjust parameters
- **Processing Pipeline**: Apply multiple processing steps in sequence
- **Export Processed Data**: Save processed patterns in various formats

### 🗄️ **Database Integration**
- Search AMCSD database by:
  - Mineral name
  - Chemical composition
  - Diffraction peaks (d-spacings)
- Download CIF (Crystallographic Information Files)
- Local database caching for offline use

### 🎯 **Phase Matching**
- **Background-Subtracted Data**: Automatically uses processed data for accurate matching
- **Theoretical Pattern Generation**: DIF file retrieval from AMCSD or CIF-based calculations
- **Advanced Peak Detection**: Improved algorithm with noise filtering and false positive reduction
- **Structure Factor Calculations**: Proper intensity calculations from crystal structures
- **Interactive Visualization**: Stick patterns with phase labels and proper scaling
- Configurable matching tolerance and scoring metrics
- Detailed match statistics and coverage analysis

### ⚙️ **Advanced Settings**
- Customizable wavelengths and X-ray sources
- Peak finding parameters
- Database connection settings
- Display preferences
- Export/import configuration

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- **PyQt5**: GUI framework (compatible with macOS)
- **matplotlib**: Scientific plotting
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Peak finding algorithms and sparse matrix operations
- **requests**: Web API access
- **beautifulsoup4**: HTML parsing
- **gemmi**: CIF file handling
- **pymatgen**: Materials science toolkit

## Usage

### Starting the Application
```bash
python main.py
```

### Basic Workflow

1. **Load Pattern Data**
   - Go to "Pattern Data" tab
   - **Drag & drop** XRD files directly into the application, or click "Load Pattern"
   - Select appropriate wavelength
   - Use "Find Peaks" to detect peak positions

2. **Process Data (Optional)**
   - Switch to "Data Processing" tab
   - **Interactive Background Subtraction**: Adjust λ (smoothness) and p (asymmetry) parameters with real-time preview
   - **Additional Processing**: Apply smoothing and noise reduction
   - **Export**: Save processed data in your preferred format

3. **Search Database**
   - Switch to "Database Search" tab
   - Choose search method (mineral name, chemistry, or peaks)
   - Enter search parameters
   - Click "Search AMCSD" to find matching phases

4. **Phase Matching**
   - Go to "Phase Matching" tab
   - Adjust tolerance and matching parameters
   - Click "Start Matching" to compare patterns
   - Review results and match statistics

5. **Configure Settings**
   - Use "Settings" tab to customize:
     - Default wavelengths
     - Database preferences
     - Matching parameters
     - Display options

### Supported File Formats

**Input Patterns:**
- `.xy` - Two-column ASCII (2θ, intensity)
- `.xye` - Three-column ASCII (2θ, intensity, error) with error bars
  - Supports C-style comments (`/* */`) and `#` comments
  - Handles multiple whitespace delimiters
  - Compatible with various XRD software output formats
- `.txt` - Tab or space-delimited text
- `.dat` - Generic data files
- `.csv` - Comma-separated values

**Database Files:**
- `.cif` - Crystallographic Information Files
- `.dif` - Diffraction data files (from AMCSD)

## X-ray Sources

### Pre-configured Wavelengths
| Source | Wavelength (Å) | Common Use |
|--------|----------------|------------|
| Cu Kα1 | 1.5406 | Most common lab source |
| Cu Kα  | 1.5418 | Average Cu radiation |
| Co Kα1 | 1.7890 | Iron-containing samples |
| Fe Kα1 | 1.9373 | Special applications |
| Cr Kα1 | 2.2897 | Long wavelength |
| Mo Kα1 | 0.7107 | High-energy source |

### Custom Wavelengths
Add your own X-ray sources in the Settings tab for specialized equipment or synchrotron radiation.

## Database Information

### AMCSD (American Mineralogist Crystal Structure Database)
- **URL**: https://rruff.geo.arizona.edu/AMS/amcsd.php
- **Content**: Crystal structures from major mineralogy journals
- **File Types**: CIF files for structures, DIF files for diffraction data
- **Coverage**: Comprehensive mineral database with >20,000 structures

### Search Methods
1. **Mineral Name**: Direct search by mineral species
2. **Chemical Elements**: Search by elemental composition
3. **Diffraction Peaks**: Match against experimental d-spacings

## Phase Matching Algorithm

The application uses an enhanced multi-step matching process:

1. **Data Processing**: Background subtraction using ALS algorithm for clean experimental data
2. **Advanced Peak Detection**: 
   - Dynamic noise level estimation
   - Prominence and width filtering
   - Local signal-to-noise analysis
   - Artifact removal (low-angle peaks < 5°)
   - Top 50 most significant peaks selection
3. **Database Query**: Retrieve candidate phases from AMCSD with CIF content
4. **Theoretical Pattern Generation**:
   - **Primary**: DIF file retrieval from AMCSD database
   - **Fallback**: Structure factor calculations from CIF data
   - Proper Bragg's law application with experimental wavelength
5. **Pattern Comparison**: Match processed experimental vs theoretical peaks
6. **Scoring**: Calculate match quality and coverage metrics

### Enhanced Features
- **Structure Factor Calculation**: F = Σ f_j × exp(2πi(hx_j + ky_j + lz_j))
- **Intensity Normalization**: |F|² with proper scaling for visualization
- **Wavelength Consistency**: Uses experimental wavelength for all calculations
- **Visual Comparison**: Stick patterns with phase identification labels

### Match Metrics
- **Match Score**: Fraction of theoretical peaks matched
- **Coverage**: Fraction of experimental peaks explained
- **d-spacing Tolerance**: Maximum allowed difference (default: 0.02 Å)
- **Intensity Correlation**: Relative intensity matching for better identification

## File Structure

```
XRDmatch/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── gui/                   # GUI components
│   ├── __init__.py
│   ├── main_window.py     # Main application window
│   ├── pattern_tab.py     # Pattern loading and analysis
│   ├── processing_tab.py  # Data processing and background subtraction
│   ├── database_tab.py    # Database search interface
│   ├── matching_tab.py    # Enhanced phase matching tools
│   └── settings_tab.py    # Configuration settings
└── utils/                 # Utility modules
    ├── __init__.py
    └── cif_parser.py      # Enhanced CIF parser with structure factors
```

## Troubleshooting

### Common Issues

**"Could not load pattern"**
- Check file format (should be 2-column numeric data)
- Ensure no header lines or use # for comments
- Verify delimiter (tab, space, or comma)

**"Search failed"**
- Check internet connection
- Verify AMCSD website is accessible
- Try reducing search scope

**"No peaks found"**
- Lower minimum height threshold in processing tab
- Apply background subtraction to improve signal-to-noise ratio
- Check if pattern has sufficient intensity above noise level
- Verify wavelength selection matches your X-ray source
- Use "Find Peaks" after processing for better results

### Performance Tips
- Use local database caching for repeated searches
- Limit search results for faster processing
- Adjust thread count in settings for your system

## Contributing

This is an open-source project. Contributions are welcome for:
- Additional database sources
- Improved matching algorithms
- New file format support
- Bug fixes and optimizations

## Citation

If you use this software in your research, please cite:

- The AMCSD database: Downs, R.T. and Hall-Wallace, M. (2003) The American Mineralogist Crystal Structure Database. American Mineralogist 88, 247-250.
- This software: XRD Phase Matcher v1.0

## License

This project is released under the MIT License. See LICENSE file for details.

## Support

For questions, bug reports, or feature requests, please create an issue in the project repository.

---

**Built with Python and PyQt5 for the crystallography community** 🔬✨
