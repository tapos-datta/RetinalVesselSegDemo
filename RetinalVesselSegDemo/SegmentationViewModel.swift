//
//  SegmentationViewModel.swift
//  RetinalVesselSegDemo
//
//  Created by Tapos Datta on 17/11/25.
//

import Foundation
import UIKit
import CoreML
import Accelerate
import CoreGraphics
import CoreImage
import CoreVideo

// MARK: - Custom Error

enum SegmentationError: Error, LocalizedError {
    case modelNotLoaded
    case noImageSelected
    case inputConversionFailed
    case predictionFailed(String)
    case outputExtractionFailed
    case maskCreationFailed
    
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "The Core ML model could not be loaded or initialized."
        case .noImageSelected: return "No image has been selected for segmentation."
        case .inputConversionFailed: return "Failed to convert the UIImage into the required MLMultiArray format."
        case .predictionFailed(let details): return "Prediction failed: \(details)"
        case .outputExtractionFailed: return "Model output 'output' was not found or is not a MultiArray."
        case .maskCreationFailed: return "Failed to create the final mask image."
        }
    }
}

// MARK: - SegmentationViewModel

@MainActor
class SegmentationViewModel: ObservableObject {
    
    // MARK: Published Properties
    @Published var originalImage: UIImage?
    @Published var segmentedImage: UIImage?
    @Published var overlayImage: UIImage?
    @Published var isProcessing = false
    @Published var errorMessage: String?
    
    // MARK: Private Properties
    // NOTE: Changed from u2net_e to VesselSegmenter to match user input
    private var model: VesselSegmenter?
    private let targetSize = CGSize(width: 384, height: 384)
    
    // Patch-wise constants
    private let patchSize: Int = 384
    private let overlap: Int = 64
    // FIX: Changed from lazy var referencing self to a direct let calculation
    // Step size is the distance between the start of consecutive patches
    private let step: Int = 384 - 64 // 320

    // MARK: Initialization
    init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try VesselSegmenter(configuration: config)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            print("Core ML Model Loading Error: \(error)")
        }
    }
    
    // MARK: Public Functions
    
    func loadImage(_ image: UIImage) {
        originalImage = image
        segmentedImage = nil
        overlayImage = nil
        errorMessage = nil
    }
    
    // The main router function
    func segmentImage() async {
        guard let image = originalImage else {
            errorMessage = SegmentationError.noImageSelected.localizedDescription
            return
        }
        
        guard model != nil else {
            errorMessage = SegmentationError.modelNotLoaded.localizedDescription
            return
        }
        
        isProcessing = true
        errorMessage = nil
        defer { isProcessing = false }
        
        let imageSize = image.size
        let shouldSegmentPatchWise = imageSize.width > targetSize.width || imageSize.height > targetSize.height

        do {
            if shouldSegmentPatchWise {
                try await segmentPatchWise()
            } else {
                try await segmentSingleImage()
            }
        } catch {
            errorMessage = error.localizedDescription
            print("Segmentation Error: \(error)")
        }
    }
    
    // MARK: - Single Image Segmentation (Non-Patch)

    private func segmentSingleImage() async throws {
        guard let image = originalImage, let model = model else { throw SegmentationError.modelNotLoaded }
        
        let inputMultiArray = try imageToMultiArray(image: image, size: targetSize)
        guard let multiArray = inputMultiArray else { throw SegmentationError.inputConversionFailed }
        
        let inputProvider = VesselSegmenterInput(input: multiArray)
        
        let outputProvider = try await Task.detached {
            return try model.prediction(input: inputProvider)
        }.value
        
        let output = outputProvider.output

        guard let maskImage = convertMultiArrayToMaskImage(output) else {
            throw SegmentationError.maskCreationFailed
        }

        let finalImage = applyMask(original: image, mask: maskImage)
        self.segmentedImage = maskImage
        self.overlayImage = finalImage
    }
    
    // MARK: - Patch-Wise Segmentation (With Overlap Blending)

    private func segmentPatchWise() async throws {
        guard let originalImage = originalImage else { throw SegmentationError.noImageSelected }
        
        // 1. Calculate and generate the full, blended mask from all patches
        let finalMaskImage = try await generateFullMask(from: originalImage)

        // 2. Apply the final mask to the original image
        guard let mask = finalMaskImage else { throw SegmentationError.maskCreationFailed }
        
        let segmented = applyMask(original: originalImage, mask: mask)
        
        // 3. Update the UI
        self.segmentedImage = finalMaskImage
        self.overlayImage = segmented
    }
    
    private func generateFullMask(from image: UIImage) async throws -> UIImage? {
        
        guard let model = model else { throw SegmentationError.modelNotLoaded }

        let width = Int(image.size.width)
        let height = Int(image.size.height)
        
        // --- FIX: Capture MainActor constants before entering nonisolated Task.detached ---
        let localPatchSize = self.patchSize
        let localStep = self.step
        let localTargetSize = self.targetSize
        // ---------------------------------------------------------------------------------
        
        // Create the final stitched arrays (one for sum of probabilities, one for pixel visit counts)
        let finalShape = [1, 1, height, width].map { NSNumber(value: $0) }
        let finalMaskSum = try MLMultiArray(shape: finalShape, dataType: .float32)
        let finalMaskCount = try MLMultiArray(shape: finalShape, dataType: .float32)

        // Run the entire complex loop on a background thread for performance
        return try await Task.detached {
            
            var startY = 0
            while startY < height {
                var startX = 0
                while startX < width {
                    
                    try autoreleasepool {
                    
                        // Use captured local constants
                        let currentPatchWidth = min(localPatchSize, width - startX)
                        let currentPatchHeight = min(localPatchSize, height - startY)

                        // 1. Crop and Pad the patch
                        let rect = CGRect(x: startX, y: startY, width: currentPatchWidth, height: currentPatchHeight)
                        
                        guard let patchImage = image.crop(to: rect),
                              let paddedPatch = patchImage.padToSquare(size: localTargetSize) else {
                            // If crop/pad fails, jump out of the autoreleasepool and continue the loop.
                            startX += localStep // Use captured localStep
                            return // Exit the autoreleasepool closure, continue the outer while loop
                        }

                        // 2. Run prediction on the 384x384 padded patch
                        let inputMultiArray = try self.imageToMultiArray(image: paddedPatch, size: localTargetSize)!
                        let outputProvider = try model.prediction(input: VesselSegmenterInput(input: inputMultiArray))
                        let outputMultiArray = outputProvider.output
                        
                        // 3. Stitch the result using averaging (blending) logic
                        self.blendOutput(
                            sourceArray: outputMultiArray,
                            sumArray: finalMaskSum,
                            countArray: finalMaskCount,
                            x: startX,
                            y: startY,
                            originalPatchWidth: currentPatchWidth,
                            originalPatchHeight: currentPatchHeight,
                            targetSize: localTargetSize
                        )
                    }
                    
                    // Move to the next patch position based on step (patchSize - overlap)
                    startX += localStep // Use captured localStep
                }
                // Move to the next row position
                startY += localStep // Use captured localStep
            }

            // 4. Final Averaging: Divide the sum of probabilities by the count of overlaps
            // Removed unnecessary 'await' before self.averageMasks()
            guard let finalAveragedMask = await self.averageMasks(sumArray: finalMaskSum, countArray: finalMaskCount) else {
                throw SegmentationError.maskCreationFailed
            }
            
            // 5. Convert the final averaged MLMultiArray to a UIImage mask
            // Removed unnecessary 'await' before self.convertMultiArrayToMaskImage()
            return self.convertMultiArrayToMaskImage(finalAveragedMask)
            
        }.value
    }
    
    // MARK: - Blending and Stitching Utilities
    
    private nonisolated func blendOutput(sourceArray: MLMultiArray, sumArray: MLMultiArray, countArray: MLMultiArray, x: Int, y: Int, originalPatchWidth: Int, originalPatchHeight: Int, targetSize: CGSize) {
        
        guard sourceArray.dataType == .float32, sumArray.dataType == .float32, countArray.dataType == .float32 else { return }

        let sourcePtr = sourceArray.dataPointer.assumingMemoryBound(to: Float.self)
        let sumPtr = sumArray.dataPointer.assumingMemoryBound(to: Float.self)
        let countPtr = countArray.dataPointer.assumingMemoryBound(to: Float.self)
        
        let destWidth = sumArray.shape.last!.intValue // Full image width
        let patchSize = Int(targetSize.width)
        
        // Only iterate over the valid (non-padded) region of the prediction
        for py in 0..<originalPatchHeight {
            for px in 0..<originalPatchWidth {
                
                // Source (Model output 384x384) index (assuming flattened HxW for the last two dimensions)
                let sourceIndex = py * patchSize + px
                
                // Destination (Full image HxW) index
                let destIndex = (y + py) * destWidth + (x + px)
                
                // Accumulate the probability value
                sumPtr[destIndex] += sourcePtr[sourceIndex]
                
                // Track how many patches contributed to this pixel
                countPtr[destIndex] += 1.0
            }
        }
    }
    
    private func averageMasks(sumArray: MLMultiArray, countArray: MLMultiArray) -> MLMultiArray? {
        
        let totalPixels = sumArray.count
        guard countArray.count == totalPixels else { return nil }
        
        let finalMask = try? MLMultiArray(shape: sumArray.shape, dataType: .float32)
        guard let finalMask = finalMask else { return nil }

        let sumPtr = sumArray.dataPointer.assumingMemoryBound(to: Float.self)
        let countPtr = countArray.dataPointer.assumingMemoryBound(to: Float.self)
        let finalPtr = finalMask.dataPointer.assumingMemoryBound(to: Float.self)
        
        // Calculate the final averaged probability for every pixel
        for i in 0..<totalPixels {
            let count = countPtr[i]
            if count > 0 {
                finalPtr[i] = sumPtr[i] / count
            } else {
                finalPtr[i] = 0.0 // Pixels that were never covered (should only be at edges of image < 384)
            }
        }
        
        return finalMask
    }
    
    // MARK: - Image Conversion Utilities

    /**
     Converts a UIImage into a normalized 1x3xHxW MLMultiArray (Channel-First layout) manually.
     
     The **Green channel value** from the image is now used to populate all three (R, G, B)
     input channels of the MLMultiArray.
     */
    private nonisolated func imageToMultiArray(image: UIImage, size: CGSize) throws -> MLMultiArray? {
        
        // Note: The sRGB -> Linear RGB conversion is handled inside imageToPixelBuffer.
        guard let pixelBuffer = imageToPixelBuffer(image: image, size: size) else { return nil }

        let width = Int(size.width)
        let height = Int(size.height)
        let channels = 3
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dataPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        // Destination MLMultiArray: 1 x 3 x H x W
        let shape = [1, channels, height, width].map { NSNumber(value: $0) }
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        let multiArrayPtr = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        
        let scaleFactor: Float = 1.0 / 255.0 // For 0-1 normalization
        
        for y in 0..<height {
            for x in 0..<width {
                // Read 4 bytes (ARGB) from the CVPixelBuffer (4 bytes per pixel)
                let pixelIndex = (y * bytesPerRow) + (x * 4)
                
                let G = dataPtr[pixelIndex + 2]
                
                let normalizedG = Float(G) * scaleFactor
                
                // Calculate the base index in the flattened HxW space
                let baseArrayOffset = y * width + x
                
                // R-Channel index (0 * H * W + baseOffset) is fed the Green channel value
                multiArrayPtr[baseArrayOffset] = normalizedG
                
                // G-Channel index (1 * H * W + baseOffset) is fed the Green channel value
                multiArrayPtr[width * height + baseArrayOffset] = normalizedG
                
                // B-Channel index (2 * H * W + baseOffset) is fed the Green channel value
                multiArrayPtr[2 * width * height + baseArrayOffset] = normalizedG
            }
        }
        
        return multiArray
    }
    
    /**
     Converts a UIImage to a CVPixelBuffer resized to a specified size, ensuring the image data is
     converted from sRGB to the **linear RGB color space** during the process.
     */
    private nonisolated func imageToPixelBuffer(image: UIImage, size: CGSize) -> CVPixelBuffer? {
        
        guard let resizedImage = image.resized(to: size) else { return nil }
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32ARGB, // Standard format for image drawing
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        // Use a Linear RGB color space for the context.
        // When the CGImage (which is usually sRGB) is drawn into this context,
        // Core Graphics automatically performs the gamma correction (sRGB -> Linear RGB).
        guard let linearRgbColorSpace = CGColorSpace(name: CGColorSpace.linearSRGB) else { return nil }

        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: linearRgbColorSpace, // Use Linear RGB Color Space
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue // ARGB order for 32ARGB pixel format
        )
        
        guard let cgImage = resizedImage.cgImage, let ctx = context else { return nil }
        
        ctx.draw(cgImage, in: CGRect(origin: .zero, size: size))
        
        return buffer
    }

    /**
     Converts the raw MLMultiArray probability output (1x1xHxW) into a binary grayscale mask UIImage.
     */
    private nonisolated func convertMultiArrayToMaskImage(_ multiArray: MLMultiArray, threshold: Float = 0.5) -> UIImage? {
        guard multiArray.dataType == .float32 else { return nil }
        
        // Assuming HxW layout for the last two dimensions
        let height = multiArray.shape[multiArray.shape.count - 2].intValue
        let width = multiArray.shape.last?.intValue ?? 0

        var maskPixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &maskPixelBuffer)
        
        guard let buffer = maskPixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let baseAddress = CVPixelBufferGetBaseAddress(buffer)
        let bufferPointer = baseAddress!.assumingMemoryBound(to: UInt8.self)
        
        let arrayPointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)
        
        for y in 0..<height {
            for x in 0..<width {
                let multiArrayIndex = (y * width) + x
                let probability = arrayPointer[multiArrayIndex]
                
                // Thresholding: Black (0) or White (255)
                let pixelValue: UInt8 = (probability > threshold) ? 255 : 0
                
                let bufferIndex = (y * CVPixelBufferGetBytesPerRow(buffer)) + (x * 4)
                
                bufferPointer[bufferIndex + 0] = pixelValue // B
                bufferPointer[bufferIndex + 1] = pixelValue // G
                bufferPointer[bufferIndex + 2] = pixelValue // R
                bufferPointer[bufferIndex + 3] = 255        // Alpha
            }
        }
        
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        
        return UIImage(cgImage: cgImage)
    }
    
    /**
     Applies the grayscale mask image as an alpha channel to the original image, creating a cutout.
     */
    private nonisolated func applyMask(original: UIImage, mask: UIImage) -> UIImage? {
        let originalSize = original.size
        
        guard let scaledMask = mask.resized(to: originalSize),
              let maskCG = scaledMask.cgImage else { return nil }
        
        guard let maskProvider = maskCG.dataProvider,
              let imageMask = CGImage(
                maskWidth: maskCG.width,
                height: maskCG.height,
                bitsPerComponent: maskCG.bitsPerComponent,
                bitsPerPixel: maskCG.bitsPerPixel,
                bytesPerRow: maskCG.bytesPerRow,
                provider: maskProvider,
                decode: nil,
                shouldInterpolate: true
              ) else { return nil }
        
        guard let originalCG = original.cgImage,
              let maskedCG = originalCG.masking(imageMask) else { return nil }
        
        return UIImage(cgImage: maskedCG)
    }
}
// MARK: - UIImage Extensions

extension UIImage {
    /**
     Resizes the image to a specified size.
     */
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    // Crops the image to the specified rectangle
    func crop(to rect: CGRect) -> UIImage? {
        guard let cgImage = self.cgImage?.cropping(to: rect) else { return nil }
        return UIImage(cgImage: cgImage, scale: self.scale, orientation: self.imageOrientation)
    }

    // Pads the image to the target size (e.g., 384x384) if it's smaller, drawing the original in the top-left corner.
    func padToSquare(size: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            // Draw image at (0, 0). The rest of the space defaults to transparent/black, which is correct for padding.
            self.draw(in: CGRect(origin: .zero, size: self.size))
        }
    }
}
