library('socialmixr')
library(data.table)
data(polymod)

ageLimits <- c(0,18,30,40,50,60,70)
ageLimits <- c(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70)

# Function to loop over country codes and save data to files
exportContactMatrices <- function(countryCodes) {
  for (code in countryCodes) {
    # Perform desired operations for each country code
    result <- contact_matrix(polymod,countries=code,age.limits=ageLimits,symmetric=TRUE)

    # Save the result to a file
    file_root <- paste0('exported_contact_data/', code)
    file_name <- paste0(file_root, ".csv")
    fwrite(result$matrix, file = file_name, sep = ",")

    # Optionally, save the other data frames as well
    fwrite(result$demography, file = paste0(file_root, "_demography.csv"), sep = ",")
    fwrite(result$participants, file = paste0(file_root, "_participants.csv"), sep = ",")

    # Add your custom code here
  }
}

# List of country codes
countryCodes <- c("BE", "DE", "FI", "GB", "IT", "LU", "NL", "PL")

# Call the function
exportContactMatrices(countryCodes)

