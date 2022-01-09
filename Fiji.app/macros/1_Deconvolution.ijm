
#@ File (label = "Input directory", style = "directory") INPUT
#@ String (label = "Deconvolution option", style = "string") Option


setBatchMode(true);


suffix = ".tif"

processFolder(INPUT);


function deconv(deconv_option){
	if(deconv_option=="A"){
		run("Colour Deconvolution", "vectors=[User values] [r1]=0.6500286 [g1]=0.704031 [b1]=0.2860126 [r2]=0.515 [g2]=0.639 [b2]=0.571 [r3]=0.268 [g3]=0.570 [b3]=0.7764");
	print("Processing: " + INPUT + File.separator + file);
	print("Saving to: " + INPUT);


	//Save the channels
   	selectWindow(title + "-(Colour_3)");
   	run("Median...", "radius=2");
	//run("Enhance Contrast...", "saturated=0.3 normalize equalize");
    run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None*");
   	saveAs("PNG",INPUT+ File.separator+replace(title,suffix,"_col1a1.png"));
	close(title + "-(Colour_3)");
	close(path);
	path = INPUT + File.separator +file;
	open(path);
    title=getTitle();


    selectWindow(title + "-(Colour_1)");
    run("Median...", "radius=2");

	run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None*");
    
    saveAs("PNG",INPUT+ File.separator+replace(title,suffix,"_hunu.png"));
    close(title + "-(Colour_1)");
    close(path);
	}
	if(deconv_option=="B"){
		run("Colour Deconvolution", "vectors=[H DAB]");
		print("Processing: " + INPUT + File.separator + file);
	print("Saving to: " + INPUT);


	//Save the channels
   	selectWindow(title + "-(Colour_2)");
   	run("Median...", "radius=2");
	//run("Enhance Contrast...", "saturated=0.3 normalize equalize");
    run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None*");
   	saveAs("PNG",INPUT+ File.separator+replace(title,suffix,"_hunu.png"));
	close(title + "-(Colour_3)");
	close(path);
	path = INPUT + File.separator +file;
	open(path);
    title=getTitle();


    selectWindow(title + "-(Colour_1)");
    run("Median...", "radius=2");

	run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None*");
    
    saveAs("PNG",INPUT+ File.separator+replace(title,suffix,"_col1a1.png"));
    close(title + "-(Colour_1)");
    close(path);
	}

}

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(INPUT) {
	list = getFileList(INPUT);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(INPUT + File.separator + list[i]))
			processFolder(INPUT  + File.separator +  list[i]);
			print(INPUT + list[i]);
			print(list[i]);
		if(endsWith(list[i], suffix))
			processFile(INPUT, list[i]);
	}
}


function processFile(INPUT, file) {
        print(file);
	path = INPUT + file;
	print(path);
	open(path);
    title=getTitle();

	deconv(Option);
	
}



