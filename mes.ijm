// Set input and output directories
inputDir = "C:/Users/Julia/PycharmProjects/imageJ/od_cut_new/";
outputDir = "C:/Users/Julia/PycharmProjects/imageJ/od_cut_new/";
thresholdValue = 40;
// Set analysis parameters
minSize = 200; // Minimum size of the colony (pixels)
maxSize = 5500000; // Maximum size of the colony (pixels)

// --- Set Measurements ---
run("Set Measurements...", "area perimeter shape integrated Feret's display add redirect=None decimal=3");



fileList = getFileList(inputDir);
for (i=1000; i<fileList.length; i++) {
    open(inputDir + fileList[i]);
    title = getTitle(); // Get the title of the current image

   // Convert to 8-bit grayscale (if needed)
    run("8-bit");
	run("Gaussian Blur...", "sigma=2");

    run("Enhance Contrast", "saturated=0.35 normalize equalize");

    setThreshold(thresholdValue, 255); // Sets the threshold range to thresholdValue - 255
    setOption("BlackBackground", true);  // Often helpful for nuclei images
    run("Convert to Mask");

    // Analyze particles
    run("Analyze Particles...", "size=" + minSize + "-" + maxSize + " circularity=0.00-1.00 show=Outlines display summarize add");
	close(title);

}

// Prepare the output file name
resultsPath = outputDir + "od.csv";

// Save the results to a CSV file
saveAs("Results", resultsPath);

print("Batch analysis complete. Results saved to: " + resultsPath);