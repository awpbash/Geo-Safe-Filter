import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  View,
  Text,
  Button,
  Image,
  StyleSheet,
  ActivityIndicator,
  Switch,
  FlatList,
  TouchableOpacity,
  Platform,
  TextInput,
  ScrollView,
  Animated,
} from "react-native";
import { launchImageLibrary } from "react-native-image-picker";
import axios from "axios";
import { Picker } from "@react-native-picker/picker";
// import { Animated } from "react-native";
import RNFS from 'react-native-fs';
import Markdown from 'react-native-markdown-display';
import { API_KEY } from '@env';
const apiKey = API_KEY;
// The server is running on port 8000 as per the Uvicorn terminal output.
const BACKEND_URL = "http://10.0.2.2:8000";

type ImageAsset = {
  uri: string;
  isRedacted: boolean;
  redactedUri: string | null;
  loading: boolean;
};

const App = () => {
  const allRedactedObjects = [
    "flag",
    "sign",
    "landmark",
    "faces",
  ];
  const [images, setImages] = useState<ImageAsset[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageAsset | null>(null);
  const [redactionMethod, setRedactionMethod] = useState<"blur" | "pixelate">(
    "blur",
  );
  const [blurKsize, setBlurKsize] = useState<string>("151");
  const [mosaicScale, setMosaicScale] = useState<string>("0.06");
  const [redactedObjects, setRedactedObjects] =
    useState<string[]>(allRedactedObjects);
  const [predictedGps, setPredictedGps] = useState<number[] | null>(null);
  const [predictedProbability, setPredictedProbability] = useState<number[] | null>(null);

  // State variables for image analysis feature
  const [riskSummary, setRiskSummary] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 400,
      useNativeDriver: true,
    }).start();
  }, [selectedImage]);

  const selectImages = useCallback(async () => {
    try {
      const result = await launchImageLibrary({
        mediaType: "photo",
        selectionLimit: 10,
      });
      if (result.assets) {
        const newImages = result.assets.map((asset) => ({
          uri: asset.uri!,
          isRedacted: false,
          redactedUri: null,
          loading: false,
        }));
        setImages(newImages);
      }
    } catch (err) {
      console.log("Image picker error: ", err);
    }
  }, []);

  // Updated `handleRedactionToggle` to accept the list of objects as a parameter
  const handleRedactionToggle = useCallback(
    async (image: ImageAsset, objectsToRedact: string[]) => {
      if (!image.redactedUri || !image.isRedacted) {
        const updatedImages = images.map((img) =>
          img.uri === image.uri
            ? { ...img, loading: true, isRedacted: true }
            : img,
        );
        setImages(updatedImages);
        setSelectedImage((prevImage) => ({
          ...prevImage!,
          loading: true,
          isRedacted: true,
        }));

        try {
          const formData = new FormData();
          formData.append("image", {
            uri: Platform.OS === "android" ? `file://${image.uri}` : image.uri,
            name: "photo.jpg",
            type: "image/jpeg",
          });

          let blurValue = parseInt(blurKsize);
          const mosaicValue = parseFloat(mosaicScale);

          if (isNaN(blurValue) || blurValue <= 0) {
            blurValue = 151;
          } else if (blurValue % 2 === 0) {
            blurValue += 1;
          }

          formData.append("method", redactionMethod);
          formData.append("blur_ksize", blurValue.toString());
          formData.append("mosaic_scale", mosaicValue.toString());
          // Serialize the array into a JSON string for a reliable payload
          formData.append("query", JSON.stringify(objectsToRedact));

          console.log("Preparing to send request with formData:");
          // Log the correct values that are being sent
          console.log("query:", JSON.stringify(objectsToRedact));

          const redactionResponse = await axios.post(
            `${BACKEND_URL}/process_image`,
            formData,
            {
              headers: {
                "Content-Type": "multipart/form-data",
              },
            },
          );

          const newRedactedUri = redactionResponse.data.redacted_image;
          console.log("Received redacted URI:", newRedactedUri);

          // Update state with the new GPS and probability data
          const { gps, probability } = redactionResponse.data.predicted_location;
          setPredictedGps(gps);
          setPredictedProbability(probability);


          const finalImages = images.map((img) =>
            img.uri === image.uri
              ? {
                ...img,
                redactedUri: newRedactedUri,
                isRedacted: true,
                loading: false,
              }
              : img,
          );
          setImages(finalImages);
          setSelectedImage((prevImage) => ({
            ...prevImage!,
            redactedUri: newRedactedUri,
            isRedacted: true,
            loading: false,
          }));
        } catch (error) {
          console.error("Redaction failed:", error);
          const finalImages = images.map((img) =>
            img.uri === image.uri ? { ...img, loading: false } : img,
          );
          setImages(finalImages);
          setSelectedImage((prevImage) => ({ ...prevImage!, loading: false }));
        }
      } else {
        const finalImages = images.map((img) =>
          img.uri === image.uri ? { ...img, isRedacted: !img.isRedacted } : img,
        );
        setImages(finalImages);
        setSelectedImage((prevImage) => ({
          ...prevImage!,
          isRedacted: !prevImage.isRedacted,
        }));
      }
    },
    [images, redactionMethod, blurKsize, mosaicScale],
  );

  const analyzeImage = useCallback(async () => {
    if (!selectedImage) {
      return;
    }

    setIsAnalyzing(true);
    setRiskSummary(null);

    // Read the file and get the Base64 data
    try {
      const base64ImageData = await RNFS.readFile(selectedImage.uri, 'base64');
      const userPrompt = "Analyze this image for any cues or risks that will make this place easily identifiable. Provide a concise summary that includes any visible personal information, dangerous objects, or identifiable locations. Recommend from list ['face', 'flag', 'landmark', sign'] on which to redact.";
      // Your API key will be here

      const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;

      const payload = {
        contents: [
          {
            role: "user",
            parts: [
              { text: userPrompt },
              {
                inlineData: {
                  mimeType: "image/jpeg",
                  data: base64ImageData // This is now the correct Base64 string
                }
              }
            ]
          }
        ],
      };

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      console.log("Image analysis response status:", response.status);

      const result = await response.json();
      const generatedText = result?.candidates?.[0]?.content?.parts?.[0]?.text;

      if (generatedText) {
        setRiskSummary(generatedText);
      } else {
        setRiskSummary("Analysis failed or no risks were identified.");
      }
    } catch (error) {
      console.error("Image analysis failed:", error);
      setRiskSummary("An error occurred during image analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedImage]);


  const renderGallery = () => (
    <Animated.View style={{ flex: 1, opacity: fadeAnim }}>
      <View style={styles.galleryContainer}>
        <Text style={styles.title}>Your Gallery</Text>
        <FlatList
          data={images}
          numColumns={2}
          keyExtractor={(item) => item.uri + "1"}
          renderItem={({ item }) => (
            <TouchableOpacity
              onPress={() => setSelectedImage(item)}
              style={styles.thumbnailContainer}
            >
              <Image source={{ uri: item.uri }} style={styles.thumbnail} />
            </TouchableOpacity>
          )}
        />
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.floatingButton}
            onPress={selectImages}
          >
            <Text style={styles.floatingButtonText}>＋</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Animated.View>
  );

  function RedactedObjectToggles({ allRedactedObjects, redactedObjects }) {
    return (
      <ScrollView contentContainerStyle={styles.toggleList}>
        {allRedactedObjects.map((object, index) => (
          <View key={index} style={styles.toggleRedactedRow}>
            <Text style={styles.toggleRedactedText}>{object}</Text>
            <Switch
              value={redactedObjects.includes(object)}
              onValueChange={() => {
                // Determine the next state
                const newRedactedObjects = redactedObjects.includes(object)
                  ? redactedObjects.filter((o) => o !== object)
                  : [...redactedObjects, object];

                // Now you can use this new array to update the state
                setRedactedObjects(newRedactedObjects);
                console.log("New Redacted Objects:", newRedactedObjects);
              }}
              thumbColor={redactedObjects.includes(object) ? "#ff2d55" : "#888"}
              trackColor={{ false: "#333", true: "#ff2d55" }}
            />
          </View>
        ))}
      </ScrollView>
    );
  }

  const renderImageView = () => (
    <ScrollView style={styles.scrollViewContainer}>
      <Animated.View style={{ flex: 1, opacity: fadeAnim }}>
        <View style={styles.imageViewerContainer}>
          <View style={styles.controls}>
            <TouchableOpacity onPress={() => setSelectedImage(null)}>
              <Text style={styles.backArrow}>&lt;</Text>
            </TouchableOpacity>

            <View style={styles.toggleRow}>
              <Text style={styles.toggleText}>mock</Text>
              {selectedImage?.loading ? (
                <ActivityIndicator size="small" color="#0000ff" />
              ) : (
                <Switch
                  value={selectedImage?.isRedacted}
                  onValueChange={() => handleRedactionToggle(selectedImage!, redactedObjects)}
                />
              )}
            </View>
          </View>

          <View style={styles.optionsContainer}>
            <Text style={styles.optionLabel}>Method:</Text>
            <View style={styles.pickerWrapper}>
              <Picker
                selectedValue={redactionMethod}
                onValueChange={(itemValue) =>
                  setRedactionMethod(itemValue)
                }
                style={styles.picker}
              >
                <Picker.Item label="Blur" value="blur" />
                <Picker.Item label="Pixelate" value="pixelate" />
              </Picker>
            </View>

            {redactionMethod === "blur" ? (
              <View style={styles.optionInput}>
                {/* <Text style={styles.optionLabel}>Blur Size:</Text>
                                <TextInput
                                    style={styles.input}
                                    onChangeText={setBlurKsize}
                                    value={blurKsize}
                                    keyboardType="numeric"
                                /> */}
              </View>
            ) : (
              <View style={styles.optionInput}>
                <Text style={styles.optionLabel}>Mosaic Scale:</Text>
                <TextInput
                  style={styles.input}
                  onChangeText={setMosaicScale}
                  value={mosaicScale}
                  keyboardType="numeric"
                />
              </View>
            )}
          </View>

          <Image
            source={{
              uri: selectedImage?.isRedacted
                ? selectedImage?.redactedUri || selectedImage.uri
                : selectedImage?.uri,
            }}
            style={styles.fullImage}
            resizeMode="cover"
          />

          <RedactedObjectToggles
            allRedactedObjects={allRedactedObjects}
            redactedObjects={redactedObjects}
          />

          {predictedGps && (
            <View style={styles.locationInfoBox}>
              <Text style={styles.locationText}>
                Predicted Location: Lat: {predictedGps[0]}, Lon: {predictedGps[1]}
              </Text>
              <Text style={styles.locationText}>
                Geo-Risk: {(predictedProbability?.[0] * 100)}%
              </Text>
              {/* <LeafletView
                                mapCenterPosition={{ lat: 1.290270, lng: 103.851959 }}
                                zoom={13}
                                mapLayers={[
                                    {
                                        url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                        attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
                                    }
                                ]}
                                onMapReady={() => console.log('Map is ready')}
                            /> */}
            </View>
          )}

          <TouchableOpacity
            style={styles.primaryButton}
            onPress={analyzeImage}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.primaryButtonText}>Analyze Image</Text>
            )}
          </TouchableOpacity>

          {riskSummary && (
           <View style={styles.summaryContainer}>
          {/* Use the Markdown component to display the text */}
          <Markdown style={markdownStyles}>
            {riskSummary}
          </Markdown>
        </View>
          )}
        </View>
      </Animated.View>
    </ScrollView>
  );

  return selectedImage ? renderImageView() : renderGallery();
};

const styles = StyleSheet.create({
  galleryContainer: {
    flex: 1,
    backgroundColor: "#000", // dark mode background
    paddingTop: 50,
    paddingHorizontal: 10,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    textAlign: "center",
    color: "#fff",
    marginBottom: 20,
  },
  thumbnailContainer: {
    flex: 1,
    margin: 3,
    borderRadius: 10,
    overflow: "hidden",
  },
  thumbnail: {
    width: "100%",
    aspectRatio: 1,
    borderRadius: 10,
  },
  buttonContainer: {
    margin: 20,
    alignItems: "center",
  },
  imageViewerContainer: {
    flex: 1,
    backgroundColor: "#000",
    paddingTop: 50,
    paddingHorizontal: 20,
    alignItems: "center",
  },
  controls: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    width: "100%",
    marginBottom: 20,
  },
  toggleRow: {
    flexDirection: "row",
    alignItems: "center",
    padding: 10,
    borderRadius: 20,
  },
  toggleText: {
    marginRight: 10,
    color: "#fff",
    fontWeight: "600",
  },
  optionsContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    flexWrap: "wrap",
    marginBottom: 20,
    gap: 10,
  },
  optionLabel: {
    marginRight: 10,
    fontWeight: "bold",
    color: "#fff",
  },
  picker: {
    height: 50,
    width: 140,
    color: "#fff",
    backgroundColor: "#1c1c1e",
    borderRadius: 10,
  },
  optionInput: {
    flexDirection: "row",
    alignItems: "center",
    marginLeft: 20,
  },
  fullImage: {
    width: "100%",
    height: 300,
    borderRadius: 10,
    marginTop: 10,
  },
  primaryButton: {
    backgroundColor: "#ff2d55", // TikTok pink
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 30,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#ff2d55",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
    marginTop: 20,
  },
  primaryButtonText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 16,
  },

  secondaryButton: {
    backgroundColor: "#1c1c1e",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryButtonText: {
    color: "#fff",
    fontSize: 14,
  },

  input: {
    height: 40,
    width: 60,
    borderColor: "#333",
    borderWidth: 1,
    borderRadius: 8,
    color: "#fff",
    backgroundColor: "#1c1c1e",
    textAlign: "center",
    fontWeight: "bold",
    fontSize: 14,
  },
  floatingButton: {
    position: "absolute",
    bottom: 30,
    alignSelf: "center",
    backgroundColor: "#ff2d55",
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#ff2d55",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 8,
  },
  floatingButtonText: {
    fontSize: 32,
    color: "#fff",
    fontWeight: "bold",
    marginTop: -2,
  },

  backArrow: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "300",
  },

  pickerWrapper: {
    backgroundColor: "#1c1c1e",
    borderRadius: 10,
    overflow: "hidden", // ensures rounding applies
    marginRight: 10,
    height: 40,
    width: 140,
    justifyContent: "center",
  },
  toggleList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    paddingHorizontal: 6,
    paddingVertical: 10,
  },
  toggleRedactedRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
    backgroundColor: "#1c1c1e",
    padding: 12,

    width: '48%',
    marginVertical: 6,
    marginHorizontal: 1,
    borderRadius: 12,
  },
  toggleRedactedText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "500",
  },
  locationInfoBox: {
    backgroundColor: "#1c1c1e",
    borderRadius: 12,
    padding: 16,
    marginTop: 20,
    width: "100%",
    alignItems: "center",
  },
  locationText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 5,
  },
  scrollViewContainer: {
    flex: 1,
    backgroundColor: "#000",
    decelerationRate: "fast",
  },
  analysisContainer: {
    backgroundColor: "#1c1c1e",
    borderRadius: 12,
    padding: 16,
    marginTop: 20,
    width: "100%",
    alignItems: "center",
  },
  analysisTitle: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 10,
  },
  analysisText: {
    color: "#fff",
    fontSize: 16,
    textAlign: 'center',
  },
  summaryContainer: {
    backgroundColor: "#1c1c1e",
    borderRadius: 12,
    padding: 16,
    marginTop: 20,
    width: "100%",
    alignItems: "center",
  }
});

const markdownStyles = {
  body: {
    color: "#fff",
    fontSize: 16,
    lineHeight: 22,
  },
  heading1: {
    color: "#ff2d55",
    fontSize: 22,
    fontWeight: "bold",
    marginBottom: 8,
  },
  heading2: {
    color: "#ff2d55",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 6,
  },
  bullet_list: {
    color: "#fff",
    marginLeft: 10,
  },
  list_item: {
    color: "#fff",
    fontSize: 16,
  },
  strong: {
    fontWeight: "bold",
    color: "#fff",
  },
  em: {
    fontStyle: "italic",
    color: "#fff",
  },
  code_inline: {
    backgroundColor: "#222",
    color: "#ff2d55",
    borderRadius: 4,
    paddingHorizontal: 4,
    fontFamily: "monospace",
  },
};

export default App;
