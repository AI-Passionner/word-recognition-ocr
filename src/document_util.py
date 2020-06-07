import subprocess 
import cv2
from PIL import Image


class DocumentUtils:
    """
    Ghostscript version 9.22 with PDF (https://www.ghostscript.com)
    usage:  
        gswin64 -q -dNOPAUSE -r300 -sDEVICE=tiffg4 -sOutputFile=b%d.tif filename.pdf -c quit
        gswin64c.exe -dNOPAUSE -r300 -sDEVICE=tiffscaled24 -sCompression=lzw -dBATCH -sOutputFile=filename.pdf
    """   
    
    @staticmethod    
    def pdf2img(pdf, out_base, resolution=None, mul_pages=False, img_type='jpg', img_format='gray'):
       
        args = ['/usr/bin/gs', "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sCompression=lzw"]
            
        if resolution is None:
            resolution = "-r" + "300"
        else:
            resolution = "-r" + str(resolution) 
            
        if img_type =='tif':   
            if img_format =='gray':                
                device = "-sDEVICE=" + "tiffscaled8"
            else:
                device = "-sDEVICE=" + "tiffscaled24"
                
        elif img_type == 'jpg':
            if img_format =='gray':                
                device = "-sDEVICE=" + "jpeggray"
            else:
                device = "-sDEVICE=" + "jpeg"
                
        elif img_type =='png':
            if img_format =='gray':                
                device = "-sDEVICE=" + "pnggray"
            else:
                device = "-sDEVICE=" + "png16m"
                
        else:
            raise Exception("Sorry, the current supporting image types {PNG, JPEG and TIF} and image format {gray and rgb}") 
           
        if mul_pages:        
            out_file = "-sOutputFile=" + out_base + "_Page_%d." + img_type 
        else:
            out_file = "-sOutputFile=" + out_base + "." + img_type
             
        args += [resolution, device, out_file, pdf, "-c quit"]     
        proc = subprocess.Popen(args, stderr=subprocess.PIPE)
        status_code, error_string = proc.wait(), proc.stderr.read()
        proc.stderr.close() 
        
        return status_code, error_string

    @staticmethod
    def img2img(input_img, out_img, out_format='PNG', quality=100):
        """
        Parameters
        ----------
        :param input_img: a string, full path the input image.
        :param out_img: a string, full path to the output image.
        :param out_format: a string, used as a saving parameter. need to be consistent to the out_img extension.
        :param quality: default 100
        Returns
        -------
        None

        """
        im = Image.open(input_img)
        page = im.convert('RGB')
        page.save(out_img, out_format, quality=quality)

        return

    @staticmethod
    def img2pdf(input_img, out_pdf):
        """Convert image to pdf"""
        return

