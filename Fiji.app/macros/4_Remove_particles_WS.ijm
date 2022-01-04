
setBatchMode(true);



//The user can either run the script via clicking Run on the IDE or by commenting 
// out (//) the line starting with #@ (line 12)and removing the // from lines starting with input (line 11) and entering 
// the path to the Deconvolved_ims. If the latter is chosen then the 
// scripts can be run via the command line (refer to the readme.md for further details)

input = "/home/atte/Documents/analysis_folder/Fiji.app/original_images/Deconvolved_ims";
//#@ File (label = "Input directory", style = "directory") input


size = "300-8500";
suffix = "Segm_TH.png";

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
			print(list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, list[i]);
	}
}
print(input);

function processFile(input, file) {
		//for (i = 0; i < 12; i++) {
	path = input + File.separator + file;
	open(path);
    title=getTitle();
	//run("Bio-Formats Importer", "open=" + path + " autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");
	
	run("Invert LUT");
	
	run("Watershed");
	run("Analyze Particles...", "size=150-4500 show=Masks");
	selectWindow("Mask of " + title);

	print("Processing: " + input + File.separator + title);
	print("Saving to: " + input);

	saveAs("PNG",input+ File.separator+replace(title,suffix,"_WS.png"));
   	
    close(path);
}


processFolder(input);

print("WATERSHED DONE!");
