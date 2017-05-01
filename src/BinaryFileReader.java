import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Created by ej on 4/30/2017.
 */
public class BinaryFileReader {

    public static byte[] readBinaryFile(String binaryFilePath) throws IOException {
        Path path = Paths.get(binaryFilePath);
        return Files.readAllBytes(path);
    }

    public static void createImgFromBytes(String path, int numImages, int imgWidth, int imgHeight){
        try {
            byte[] data0bytes = readBinaryFile(path);

            for (int imgOffset = 0; imgOffset < numImages; imgOffset++){
                BufferedImage res = new BufferedImage( imgWidth, imgHeight, BufferedImage.TYPE_INT_RGB );

                for (int index = 0; index < imgWidth * imgHeight; index++){

                    int x = index % imgWidth;
                    int y = index / imgHeight;

                    int i = index + (imgOffset * imgHeight * imgWidth);
                    int value = data0bytes[i] & 0xff;

                    res.setRGB(x, y, value);
                }
                ImageIO.write(res, "bmp", new File("output" + imgOffset + ".bmp"));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
