# NYC Payroll Data Cleaning Summary

**Input File:** Citywide_Payroll_Data__Fiscal_Year__20251009.csv (NYC Open Data)  
**Output File:** nyc_annual_salary_employees_payBasis_perAnuum.csv  
**Author:** Mehreen Ali Gillani

---

## Overview
This document summarizes the data cleaning pipeline applied to NYC Payroll data containing ~6.7 million employee records. The cleaning process focuses on data validation, standardization, and filtering for analysis-ready datasets.

---

## 1. **Initial Data Assessment**
- **Dataset Size:** 6.7 million records with 9 text columns and multiple salary/pay columns
- **Key Numeric Columns:** Base Salary, Regular Gross Paid, OT Paid, Total Other Pay
- **Key Categorical Columns:** Agency Name, Title Description, Leave Status, Work Location Borough
- **Data Issues Identified:**
  - Date columns in object format requiring conversion
  - Missing values in several categorical columns
  - Negative and zero values in salary/pay columns
  - Non-NYC locations mixed with NYC boroughs

---

## 2. **Duplicate Removal**
- **Action:** Removed exact duplicate rows
- **Result:** Reduced from original count to clean dataset by removing redundant employee records

---

## 3. **Date Processing - Agency Start Date**
- **Conversion:** Converted 'Agency Start Date' from MM/DD/YYYY format to datetime
- **Validation:** Clipped dates to valid range (1970-01-01 to 2025-10-31)
- **Handling:** Rows with invalid dates converted to NaT (Not a Time)
- **Output:** New column `Employee_Agency_Tenure` calculated in years as of June 30 of fiscal year
  - Formula: (June 30 of fiscal year - Agency Start Date) / 365.25
  - Negative values clipped to 0

---

## 4. **Leave Status Filtering**
- **Original Distribution:** Checked all leave status values
- **Filter Applied:** Kept only "ACTIVE" employees
- **Rationale:** Focused analysis on current workforce, excluded terminated/inactive employees

---

## 5. **Geographic Filtering - Work Location Borough**
- **Initial Issue:** Mix of valid NYC boroughs, non-NYC locations, and missing values
- **Standardization Mapping:**
  - BRONX → BRONX
  - BROOKLYN → BROOKLYN
  - MANHATTAN → MANHATTAN
  - QUEENS → QUEENS
  - RICHMOND → STATEN ISLAND (official name correction)
  - Various case variations (lowercase) → standardized uppercase
  
- **Deletion:** Removed all non-NYC location rows (where borough mapped to NaN)
- **Result:** Retained only employees in 5 official NYC boroughs

---

## 6. **Salary/Pay Validation**
- **Identified Issues:**
  - Negative values in Base Salary, Regular Gross Paid, Total OT Paid, Total Other Pay
  - Zero values in salary columns
  
- **Action:** Deleted all rows containing ANY negative values in pay columns
- **Justification:** Negative/invalid pay data unsuitable for analysis

---

## 7. **Hours Validation**
- **Columns Checked:** Regular Hours and OT (Overtime) Hours
- **Action:** Removed rows where either column had negative values
- **Kept:** Rows with 0 or positive values

---

## 8. **Data Type Optimization**
- **Regular Hours & OT Hours:** Converted to `float32`
- **Pay Columns:** Converted to `float32`
  - Base Salary
  - Regular Gross Paid
  - Total OT Paid
  - Total Other Pay
- **Fiscal Year:** Converted to `Int64`
- **Benefit:** Reduced memory footprint while maintaining precision

---

## 9. **Column Removal**
- **Dropped Column:** 'Mid Init' (>2.7 million missing values - ~40% of dataset)
- **Rationale:** Too sparse to be useful for analysis

---

## 10. **Agency Standardization**
- **Approach:** Created function `standardize_agency()` to clean inconsistent agency names
- **Key Mappings:**
  - Education variations → 'Department of Education'
  - Police keywords → 'Police Department'
  - Fire keywords → 'Fire Department'
  - Sanitation → 'Department of Sanitation'
  - Correction → 'Department of Correction'
  - Health → 'Department of Health'
  - Parks → 'Department of Parks & Recreation'
  - Other agencies standardized similarly
  
- **Result:** Consolidated ~150+ agency variations into standardized categories
- **Output:** New column `Agency_Std`

---

## 11. **Title Categorization - Broad Categories**
- **Function:** `create_broad_title_category()` groups detailed titles into 8 broad categories
- **Categories Created:**
  1. Police
  2. Fire Department
  3. Sanitation
  4. Corrections
  5. Education - Teaching
  6. Education - Support
  7. Medical/Health
  8. Administrative
  9. Management
  10. Technical
  11. Other Roles

- **Output:** New column `Title_Category`

---

## 12. **Title Standardization - Detailed Titles**
- **Function:** `standardize_title_description()` normalizes individual job titles
- **Coverage:** Standardized ~100+ common titles including:
  - **Education:** Teacher, Principal, Para-Professional, School Secretary, etc.
  - **Police:** Police Officer, Detective, Sergeant, Lieutenant, Captain, etc.
  - **Fire:** Firefighter
  - **Corrections:** Correction Officer
  - **Medical:** Nurse, Doctor, EMT, Paramedic
  - **Administrative:** Clerical Associate, Administrative Analyst, etc.
  
- **Reduction:** From 10,000+ original titles to ~300 standardized titles
- **Output:** New column `Title_Std`

---

## 13. **Pay Basis Filtering**
- **Original Distributions:** Multiple pay basis types (Hourly, per Annum, etc.)
- **Filter Applied:** Kept only employees with "per Annum" pay basis
- **Rationale:** Focuses analysis on salaried employees for consistency

---

## 14. **Fiscal Year Filtering**
- **Removed 2014:** Incomplete data for first year
- **Removed 2025:** Incomplete/partial year data
- **Final Timeframe:** 2015-2024 (10 full fiscal years)
- **Rationale:** Ensures complete, comparable annual data

---

## 15. **Final Dataset Statistics**
- **Total Rows:** Final dataset contains salaried NYC employees across 2015-2024
- **Total Columns:** Reduced from original (dropped Mid Init, added derived columns)
- **Data Ranges (all positive):**
  - Regular Hours: 0.00 - max hours worked
  - OT Hours: 0.00 - max overtime hours
  - Base Salary: minimum to maximum salary
  
- **Geographic Coverage:** 5 NYC boroughs
- **Time Coverage:** 10 fiscal years (2015-2024)

---

## 16. **Output File**
- **File Name:** nyc_annual_salary_employees_payBasis_perAnuum.csv
- **Format:** CSV with all cleaned and standardized columns
- **Quality:** Analysis-ready dataset with validated data types and cleaned values

---

## Key Quality Metrics
| Metric | Result |
|--------|--------|
| Duplicate Rows Removed | Yes |
| Invalid Dates Handled | Yes |
| Negative Pay Values Removed | Yes |
| Non-NYC Locations Removed | Yes |
| Data Types Optimized | Yes |
| Agency Names Standardized | Yes |
| Job Titles Standardized | Yes |
| Missing Leave Status Filtered | Yes |
| Incomplete Years Removed | Yes |

---

## Visualization Generated
- **Chart Type:** Line chart
- **Title:** "NYC Salaried Workforce Trend: 2015-2024"
- **Features:**
  - Shows employee count per fiscal year
  - Highlights pandemic period (2019.5-2020.5)
  - Includes pre/post-pandemic annotations
  - Shows peak and low values
  - Displays data source attribution
  
**Key Insights:**
- Overall workforce trend (growth/stability)
- Pandemic impact on employment
- Recent hiring/retention trends
