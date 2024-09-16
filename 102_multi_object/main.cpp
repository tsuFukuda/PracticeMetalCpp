//
//  main.cpp
//  102_multi_object
//
//  Created by FT on 2024/09/16.
//

#include <cassert>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>
#include <simd/simd.h>

static constexpr size_t kMaxFramesInFlight = 3;
static constexpr int kSquareVertexNum = 4;
static constexpr int kSquareDivVertexNum = 6;
static constexpr int kCubeVertexNum = 8;
static constexpr int kCubeDivVertexNum = 36;

#pragma region Declarations {


class Renderer {
public:
    Renderer(MTL::Device* pDevice);
    ~Renderer();
    
    void buildShaders();
    void buildBuffers();
    void draw(MTK::View* pView);
    void buildDepthStencilStates();

    void buildSquareBuffers();
    void buildCubeBuffers();

private:
    MTL::Device* _pDevice;
    MTL::CommandQueue* _pCommandQueue;
    MTL::RenderPipelineState* _pPSO;
    MTL::DepthStencilState* _pDepthStencilState;
    
    MTL::Buffer* _pInstanceDataBuffer[kMaxFramesInFlight];
    MTL::Buffer* _pCameraDataBuffer[kMaxFramesInFlight];
    
    MTL::Buffer* _pSquareVertexPositionsBuffer;
    MTL::Buffer* _pSquareVertexColorsBuffer;
    MTL::Buffer* _pSquareIndicesBuffer;
    
    MTL::Buffer* _pCubeVertexPositionsBuffer;
    MTL::Buffer* _pCubeVertexColorsBuffer;
    MTL::Buffer* _pCubeIndicesBuffer;
    
    int _frame;
    dispatch_semaphore_t _semaphore;
};

class MyMTKViewDelegate : public MTK::ViewDelegate {
public:
    MyMTKViewDelegate(MTL::Device* pDevice);
    virtual ~MyMTKViewDelegate() override;
    virtual void drawInMTKView(MTK::View* pView) override;

private:
    Renderer* _pRenderer;
};

class MyAppDelegate : public NS::ApplicationDelegate {
public:
    ~MyAppDelegate();

    NS::Menu* createMenuBar();

    virtual void applicationWillFinishLaunching(NS::Notification* pNotification) override;
    virtual void applicationDidFinishLaunching(NS::Notification* pNotification) override;
    virtual bool applicationShouldTerminateAfterLastWindowClosed(NS::Application* pSender) override;

private:
    NS::Window* _pWindow;
    MTK::View* _pMtkView;
    MTL::Device* _pDevice;
    MyMTKViewDelegate* _pViewDelegate = nullptr;
};

#pragma endregion Declarations }


int main(int argc, char* argv[]) {
    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    MyAppDelegate del;

    NS::Application* pSharedApplication = NS::Application::sharedApplication();
    pSharedApplication->setDelegate(&del);
    pSharedApplication->run();

    pAutoreleasePool->release();

    return 0;
}


#pragma mark - AppDelegate
#pragma region AppDelegate {

MyAppDelegate::~MyAppDelegate() {
    _pMtkView->release();
    _pWindow->release();
    _pDevice->release();
    delete _pViewDelegate;
}

NS::Menu* MyAppDelegate::createMenuBar() {
    using NS::StringEncoding::UTF8StringEncoding;

    NS::Menu* pMainMenu = NS::Menu::alloc()->init();
    NS::MenuItem* pAppMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pAppMenu = NS::Menu::alloc()->init(NS::String::string("Appname", UTF8StringEncoding));

    NS::String* appName = NS::RunningApplication::currentApplication()->localizedName();
    NS::String* quitItemName = NS::String::string("Quit ", UTF8StringEncoding)->stringByAppendingString(appName);
    SEL quitCb = NS::MenuItem::registerActionCallback("appQuit", [](void*, SEL, const NS::Object* pSender) {
        auto pApp = NS::Application::sharedApplication();
        pApp->terminate(pSender);
    });

    NS::MenuItem* pAppQuitItem = pAppMenu->addItem(quitItemName, quitCb, NS::String::string("q", UTF8StringEncoding));
    pAppQuitItem->setKeyEquivalentModifierMask(NS::EventModifierFlagCommand);
    pAppMenuItem->setSubmenu(pAppMenu);

    NS::MenuItem* pWindowMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pWindowMenu = NS::Menu::alloc()->init(NS::String::string("Window", UTF8StringEncoding));

    SEL closeWindowCb = NS::MenuItem::registerActionCallback("windowClose", [](void*, SEL, const NS::Object*) {
        auto pApp = NS::Application::sharedApplication();
            pApp->windows()->object<NS::Window>(0)->close();
    });
    NS::MenuItem* pCloseWindowItem = pWindowMenu->addItem(
        NS::String::string("Close Window", UTF8StringEncoding),
        closeWindowCb,
        NS::String::string("w", UTF8StringEncoding));
    pCloseWindowItem->setKeyEquivalentModifierMask(NS::EventModifierFlagCommand);

    pWindowMenuItem->setSubmenu(pWindowMenu);

    pMainMenu->addItem(pAppMenuItem);
    pMainMenu->addItem(pWindowMenuItem);

    pAppMenuItem->release();
    pWindowMenuItem->release();
    pAppMenu->release();
    pWindowMenu->release();

    return pMainMenu->autorelease();
}

void MyAppDelegate::applicationWillFinishLaunching(NS::Notification* pNotification) {
    NS::Menu* pMenu = createMenuBar();
    NS::Application* pApp = reinterpret_cast<NS::Application*>(pNotification->object());
    pApp->setMainMenu(pMenu);
    pApp->setActivationPolicy(NS::ActivationPolicy::ActivationPolicyRegular);
}

void MyAppDelegate::applicationDidFinishLaunching(NS::Notification* pNotification) {
    CGRect frame = (CGRect){{100.0, 100.0}, {512.0, 512.0}};

    _pWindow = NS::Window::alloc()->init(
        frame,
        NS::WindowStyleMaskClosable|NS::WindowStyleMaskTitled,
        NS::BackingStoreBuffered,
        false);

    _pDevice = MTL::CreateSystemDefaultDevice();

    _pMtkView = MTK::View::alloc()->init(frame, _pDevice);
    _pMtkView->setColorPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
    _pMtkView->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 0.0));

    _pViewDelegate = new MyMTKViewDelegate(_pDevice);
    _pMtkView->setDelegate(_pViewDelegate);

    _pWindow->setContentView(_pMtkView);
    _pWindow->setTitle(NS::String::string("101 - Square", NS::StringEncoding::UTF8StringEncoding));

    _pWindow->makeKeyAndOrderFront(nullptr);

    NS::Application* pApp = reinterpret_cast<NS::Application*>(pNotification->object());
    pApp->activateIgnoringOtherApps(true);
}

bool MyAppDelegate::applicationShouldTerminateAfterLastWindowClosed(NS::Application* pSender) {
    return true;
}

#pragma endregion AppDelegate }


#pragma mark - ViewDelegate
#pragma region ViewDelegate {

MyMTKViewDelegate::MyMTKViewDelegate(MTL::Device* pDevice)
    : MTK::ViewDelegate(), _pRenderer(new Renderer(pDevice)) {
}

MyMTKViewDelegate::~MyMTKViewDelegate() {
    delete _pRenderer;
}

void MyMTKViewDelegate::drawInMTKView(MTK::View* pView) {
    _pRenderer->draw(pView);
}

#pragma endregion ViewDelegate }

#pragma mark - Math
namespace math {

constexpr simd::float3 add(const simd::float3& a, const simd::float3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

constexpr simd_float4x4 makeIdentity() {
    return (simd_float4x4){
        (simd::float4){ 1.f, 0.f, 0.f, 0.f },
        (simd::float4){ 0.f, 1.f, 0.f, 0.f },
        (simd::float4){ 0.f, 0.f, 1.f, 0.f },
        (simd::float4){ 0.f, 0.f, 0.f, 1.f },};
}

simd::float4x4 makePerspective(float fovRadians, float aspect, float znear, float zfar) {
    using simd::float4;
    float ys = 1.f / tanf(fovRadians * 0.5f);
    float xs = ys / aspect;
    float zs = zfar / ( znear - zfar );
    return simd_matrix_from_rows((simd::float4){xs, 0.0f, 0.0f, 0.0f},
                                 (simd::float4){0.0f, ys, 0.0f, 0.0f},
                                 (simd::float4){0.0f, 0.0f, zs, znear * zs},
                                 (simd::float4){0, 0, -1, 0});
}

simd::float4x4 makeTranslate(const simd::float3& v) {
    const simd::float4 col0 = {1.0f, 0.0f, 0.0f, 0.0f};
    const simd::float4 col1 = {0.0f, 1.0f, 0.0f, 0.0f};
    const simd::float4 col2 = {0.0f, 0.0f, 1.0f, 0.0f};
    const simd::float4 col3 = {v.x, v.y, v.z, 1.0f};
    return simd_matrix(col0, col1, col2, col3);
}

simd::float4x4 makeScale(const simd::float3& v) {
    return simd_matrix((simd::float4){v.x, 0, 0, 0},
                       (simd::float4){0, v.y, 0, 0},
                       (simd::float4){0, 0, v.z, 0},
                       (simd::float4){0, 0, 0, 1.0});
}



}


#pragma mark - Renderer
#pragma region Renderer {

Renderer::Renderer(MTL::Device* pDevice)
    : _pDevice(pDevice->retain()), _frame(0) {
    _pCommandQueue = _pDevice->newCommandQueue();
    buildShaders();
    buildDepthStencilStates();
    buildBuffers();
        
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
}

Renderer::~Renderer() {
    _pDepthStencilState->release();
    _pCubeVertexPositionsBuffer->release();
    _pCubeVertexColorsBuffer->release();
    _pSquareVertexPositionsBuffer->release();
    _pSquareVertexColorsBuffer->release();
    _pPSO->release();
    _pCommandQueue->release();
    _pDevice->release();
}

namespace shader_types
{
    struct InstanceData
    {
        simd::float4x4 instanceTransform;
        simd::float4 instanceColor;
    };

    struct CameraData
    {
        simd::float4x4 perspectiveTransform;
        simd::float4x4 worldTransform;
    };
}

void Renderer::buildShaders() {
    using NS::StringEncoding::UTF8StringEncoding;

    const char* shaderSrc = R"(
        #include <metal_stdlib>
        using namespace metal;
    
        struct v2f {
            float4 position [[position]];
            half3 color;
        };

        struct VertexData {
            float3 position;
        };

        struct InstanceData {
            float4x4 instanceTransform;
            float4 instanceColor;
        };

        struct CameraData {
            float4x4 perspectiveTransform;
            float4x4 worldTransform;
        };

        v2f vertex vertexMain(
                device const VertexData* vertexData [[buffer(0)]],
                device const InstanceData* instanceData [[buffer(1)]],
                device const CameraData& cameraData [[buffer(2)]],
                uint vertexId [[vertex_id]],
                uint instanceId [[instance_id]]) {
            v2f o;
            float4 pos = float4(vertexData[vertexId].position, 1.0);
            pos = instanceData[instanceId].instanceTransform * pos;
            pos = cameraData.perspectiveTransform * cameraData.worldTransform * pos;
            o.position = pos;
            o.color = half3(instanceData[instanceId].instanceColor.rgb);
            return o;
        }

        half4 fragment fragmentMain(v2f in [[stage_in]]) {
            return half4(in.color, 1.0);
        }
    )";

    NS::Error* pError = nullptr;
    MTL::Library* pLibrary = _pDevice->newLibrary(NS::String::string(shaderSrc, UTF8StringEncoding), nullptr, &pError);
    if (!pLibrary) {
        __builtin_printf("%s", pError->localizedDescription()->utf8String());
        assert(false);
    }

    MTL::Function* pVertexFn = pLibrary->newFunction(NS::String::string("vertexMain", UTF8StringEncoding));
    MTL::Function* pFragFn = pLibrary->newFunction(NS::String::string("fragmentMain", UTF8StringEncoding));

    MTL::RenderPipelineDescriptor* pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pDesc->setVertexFunction(pVertexFn);
    pDesc->setFragmentFunction(pFragFn);
    pDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);

    _pPSO = _pDevice->newRenderPipelineState(pDesc, &pError);
    if (!_pPSO) {
        __builtin_printf("%s", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pVertexFn->release();
    pFragFn->release();
    pDesc->release();
    pLibrary->release();
}

void Renderer::buildDepthStencilStates() {
    MTL::DepthStencilDescriptor* pDsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    pDsDesc->setDepthCompareFunction(MTL::CompareFunction::CompareFunctionLess);
    pDsDesc->setDepthWriteEnabled(true);

    _pDepthStencilState = _pDevice->newDepthStencilState(pDsDesc);

    pDsDesc->release();
}

void Renderer::buildBuffers() {
    buildSquareBuffers();
    buildCubeBuffers();
}

void Renderer::buildSquareBuffers() {
    const float s = 0.5f;
    simd::float3 positions[kSquareVertexNum] = {
        {-s, -s, 0.0f},
        {+s, -s, 0.0f},
        {+s, +s, 0.0f},
        {-s, +s, 0.0f},
    };
    simd::float3 colors[kSquareVertexNum] = {
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
    };
    uint16_t indices[kSquareDivVertexNum] = {
        0, 1, 2,
        2, 3, 0,
    };
    
    const size_t positionDataSize = sizeof(positions);
    const size_t colorDataSize = sizeof(colors);
    const size_t indexDataSize = sizeof(indices);
    
    MTL::Buffer* pVertexPositionsBuffer = _pDevice->newBuffer(sizeof(positions), MTL::ResourceStorageModeManaged);
    MTL::Buffer* pVertexColorsBuffer = _pDevice->newBuffer(sizeof(colors), MTL::ResourceStorageModeManaged);
    MTL::Buffer* pIndicesBuffer = _pDevice->newBuffer(sizeof(indices), MTL::ResourceStorageModeManaged);
    
    _pSquareVertexPositionsBuffer = pVertexPositionsBuffer;
    _pSquareVertexColorsBuffer = pVertexColorsBuffer;
    _pSquareIndicesBuffer = pIndicesBuffer;
    
    memcpy(_pSquareVertexPositionsBuffer->contents(), positions, positionDataSize);
    memcpy(_pSquareVertexColorsBuffer->contents(), colors, colorDataSize);
    memcpy(_pSquareIndicesBuffer->contents(), indices, indexDataSize);
    
    _pSquareVertexPositionsBuffer->didModifyRange(NS::Range::Make(0, _pSquareVertexPositionsBuffer->length()));
    _pSquareVertexColorsBuffer->didModifyRange(NS::Range::Make(0, _pSquareVertexColorsBuffer->length()));
    _pSquareIndicesBuffer->didModifyRange(NS::Range::Make(0, _pSquareIndicesBuffer->length()));
}

void Renderer::buildCubeBuffers() {
    const float s = 0.5f;
    simd::float3 positions[kCubeVertexNum] = {
        {-s, -s, +s},
        {+s, -s, +s},
        {+s, +s, +s},
        {-s, +s, +s},
        {-s, -s, -s},
        {-s, +s, -s},
        {+s, +s, -s},
        {+s, -s, -s},
    };
    simd::float3 colors[kCubeVertexNum] = {
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
    };
    uint16_t indices[kCubeDivVertexNum] = {
        // 前面
        0, 1, 2,
        2, 3, 0,
        // 右面
        1, 7, 6,
        6, 2, 1,
        // 背面
        7, 4, 5,
        5, 6, 7,
        // 左面
        4, 0, 3,
        3, 5, 4,
        // 上面
        3, 2, 6,
        6, 5, 3,
        // 下面
        4, 7, 1,
        1, 0, 4,
    };
    
    const size_t positionDataSize = sizeof(positions);
    const size_t colorDataSize = sizeof(colors);
    const size_t indexDataSize = sizeof(indices);
    
    MTL::Buffer* pVertexPositionsBuffer = _pDevice->newBuffer(sizeof(positions), MTL::ResourceStorageModeManaged);
    MTL::Buffer* pVertexColorsBuffer = _pDevice->newBuffer(sizeof(colors), MTL::ResourceStorageModeManaged);
    MTL::Buffer* pIndicesBuffer = _pDevice->newBuffer(sizeof(indices), MTL::ResourceStorageModeManaged);
    
    _pCubeVertexPositionsBuffer = pVertexPositionsBuffer;
    _pCubeVertexColorsBuffer = pVertexColorsBuffer;
    _pCubeIndicesBuffer = pIndicesBuffer;
    
    memcpy(_pCubeVertexPositionsBuffer->contents(), positions, positionDataSize);
    memcpy(_pCubeVertexColorsBuffer->contents(), colors, colorDataSize);
    memcpy(_pCubeIndicesBuffer->contents(), indices, indexDataSize);
    
    _pCubeVertexPositionsBuffer->didModifyRange(NS::Range::Make(0, _pCubeVertexPositionsBuffer->length()));
    _pCubeVertexColorsBuffer->didModifyRange(NS::Range::Make(0, _pCubeVertexColorsBuffer->length()));
    _pCubeIndicesBuffer->didModifyRange(NS::Range::Make(0, _pCubeIndicesBuffer->length()));
}

void Renderer::draw(MTK::View* pView) {
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    
    _frame = (_frame + 1) % kMaxFramesInFlight;
    MTL::Buffer* pInstanceDataBuffer = _pInstanceDataBuffer[_frame];

    MTL::CommandBuffer* pCmd = _pCommandQueue->commandBuffer();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    Renderer* pRenderer = this;
    pCmd->addCompletedHandler(^void(MTL::CommandBuffer* pCmd){
        dispatch_semaphore_signal(pRenderer->_semaphore);
    });
    shader_types::InstanceData* pInstanceData = reinterpret_cast<shader_types::InstanceData*>(pInstanceDataBuffer->contents());
    
    // 四角形
    {
        simd::float3 objectPosition = {0.0f, 0.0f, 0.0f};
        simd::float4x4 translate = math::makeTranslate(math::add(objectPosition, {0.5f, 0.5f, 0.f}));
        pInstanceData[0].instanceTransform = translate;
        pInstanceData[0].instanceColor = (simd::float4){0.0f, 1.0f, 0.0f, 1.0f};
    }
    // 立方体
    {
        simd::float3 objectPosition = {0.0f, 0.0f, 0.0f};
        simd::float4x4 translate = math::makeTranslate(math::add(objectPosition, {-0.5f, -0.5f, 0.f}));
        pInstanceData[1].instanceTransform = translate;
        pInstanceData[1].instanceColor = (simd::float4){0.0f, 1.0f, 0.0f, 1.0f};
    }
    pInstanceDataBuffer->didModifyRange(NS::Range::Make(0, pInstanceDataBuffer->length()));
    
    MTL::Buffer* pCameraDataBuffer = _pCameraDataBuffer[_frame];
    shader_types::CameraData* pCameraData = reinterpret_cast<shader_types::CameraData*>(pCameraDataBuffer->contents());
    pCameraData->perspectiveTransform = math::makePerspective(45.f * M_PI / 180.f, 1.f, 0.03f, 500.0f) ;
    pCameraData->worldTransform = math::makeIdentity();
    pCameraDataBuffer->didModifyRange(NS::Range::Make(0, sizeof(shader_types::CameraData)));
    
    MTL::RenderPassDescriptor* pRpd = pView->currentRenderPassDescriptor();
    MTL::RenderCommandEncoder* pEnc = pCmd->renderCommandEncoder(pRpd);

    pEnc->setRenderPipelineState( _pPSO );
    pEnc->setDepthStencilState(_pDepthStencilState);

    // TODO
    /*
    pEnc->setVertexBuffer( _pVertexDataBuffer, 0, 0 );
    pEnc->setVertexBuffer( pInstanceDataBuffer, 0, 1);
    pEnc->setVertexBuffer( pCameraDataBuffer, 0, 2 );

    pEnc->setCullMode( MTL::CullModeBack );
    pEnc->setFrontFacingWinding( MTL::Winding::WindingCounterClockwise );

    pEnc->drawIndexedPrimitives( MTL::PrimitiveType::PrimitiveTypeTriangle,
                                6 * 6, MTL::IndexType::IndexTypeUInt16,
                                _pIndexBuffer,
                                0,
                                kNumInstances );
    */

    pEnc->endEncoding();
    pCmd->presentDrawable( pView->currentDrawable() );
    pCmd->commit();

    pPool->release();
}

#pragma endregion Renderer }
