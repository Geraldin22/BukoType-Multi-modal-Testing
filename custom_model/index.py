import argparse
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO Live Webcam Detection")
    parser.add_argument('--model', default='train/weights/best.pt')
    parser.add_argument('--source', default='0')
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)

    print("=== BINUBUKSAN ANG LIVE CAMERA POP-UP WINDOW ===")
    print("I-click ang window at pindutin ang 'q' o i-Ctrl+C sa terminal para isara.")

    # Gumagamit ng stream=True para sa tuluy-tuloy na video frames
    for result in model.predict(source=source, show=False, stream=True):
        # Kukunin natin ang imahe na may nakaguhit nang bounding boxes mula sa YOLO
        frame = result.plot()
        
        # Ito ang magpapalabas ng pop-up window sa Windows mo!
        cv2.imshow("YOLO Live Detection - Laptop Cam", frame)
        
        # Hihinto ang loop kapag pinindot ang letrang 'q' sa keyboard habang nasa window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
