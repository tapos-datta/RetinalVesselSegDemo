//
//  MainView.swift
//  RetinalVesselSegDemo
//
//  Created by Tapos Datta on 16/11/25.
//

import SwiftUI
import PhotosUI
import CoreML


struct MainView: View {
    @StateObject private var viewModel = SegmentationViewModel()
    @State private var selectedItem: PhotosPickerItem?
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Title
                    Text("Retinal Vessel Segmentation")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .padding(.top)
                    
                    // Image picker button
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Select Retinal Image", systemImage: "photo.on.rectangle")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                    }
                    .onChange(of: selectedItem) { newItem in
                        Task {
                            if let data = try? await newItem?.loadTransferable(type: Data.self),
                               let uiImage = UIImage(data: data) {
                                viewModel.loadImage(uiImage)
                            }
                        }
                    }
                    
                    // Original image
                    if let originalImage = viewModel.originalImage {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Original Image")
                                .font(.headline)
                            Image(uiImage: originalImage)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 300)
                                .cornerRadius(12)
                                .shadow(radius: 5)
                        }
                    }
                    
                    // Segment button
                    if viewModel.originalImage != nil && !viewModel.isProcessing {
                        Button(action: {
                            Task {
                                await viewModel.segmentImage()
                            }
                        }) {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                } else {
                                    Image(systemName: "wand.and.stars")
                                }
                                Text(viewModel.isProcessing ? "Processing..." : "Segment Vessels")
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isProcessing ? Color.gray : Color.green)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isProcessing)
                    }
                    
                    // Segmented result
                    if let segmentedImage = viewModel.segmentedImage {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Segmented Vessels")
                                .font(.headline)
                            Image(uiImage: segmentedImage)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 300)
                                .cornerRadius(12)
                                .shadow(radius: 5)
                        }
                    }
                    
                    // Overlay view (optional)
                    if let _ = viewModel.originalImage,
                        let _ = viewModel.segmentedImage {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Overlay")
                                .font(.headline)
                            Image(uiImage: viewModel.overlayImage!)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 300)
                                .cornerRadius(12)
                                .shadow(radius: 5)
                        }
                    }
                    
                    // Error message
                    if let errorMessage = viewModel.errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .padding()
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(8)
                    }
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

#Preview {
    MainView()
}
