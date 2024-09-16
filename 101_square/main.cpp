//
//  main.cpp
//  101_square
//
//  Created by FT on 2024/09/14.
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

#define POLYGON_INDICES (4)
#define DIV_POLYGON_INDICES ((POLYGON_INDICES - 2) * 3)

#pragma region Declarations {

class Renderer {
    public:
        Renderer(MTL::Device* pDevice);
        ~Renderer();
    
        void buildShaders();
        void buildBuffers();
        void draw(MTK::View* pView);

    private:
        MTL::Device* _pDevice;
        MTL::CommandQueue* _pCommandQueue;
        MTL::RenderPipelineState* _pPSO;
        MTL::Buffer* _pVertexPositionsBuffer;
        MTL::Buffer* _pVertexColorsBuffer;
        MTL::Buffer* _pIndicesBuffer;
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


#pragma mark - Renderer
#pragma region Renderer {

Renderer::Renderer(MTL::Device* pDevice) : _pDevice(pDevice->retain()) {
    _pCommandQueue = _pDevice->newCommandQueue();
    buildShaders();
    buildBuffers();
}

Renderer::~Renderer() {
    _pVertexPositionsBuffer->release();
    _pVertexColorsBuffer->release();
    _pPSO->release();
    _pCommandQueue->release();
    _pDevice->release();
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

        v2f vertex vertexMain(
                uint vertexId [[vertex_id]],
                device const float3* positions [[buffer(0)]],
                device const float3* colors [[buffer(1)]]) {
            v2f o;
            o.position = float4(positions[vertexId], 1.0);
            o.color = half3 (colors[vertexId]);
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

void Renderer::buildBuffers() {
    const float s = 0.5f;
    
    simd::float3 positions[POLYGON_INDICES] = {
        {-s, -s, 0.0f},
        {+s, -s, 0.0f},
        {+s, +s, 0.0f},
        {-s, +s, 0.0f},
    };
    // グラデーションをつけるために3つ目の色データをRに変更すると、赤の斜め線が入ってしまう。
    // 三角形に分割して描画していることに起因すると思われるが、解消するためにはどのようにすれば良いのだろうか。
    // 頂点・色データを6つに増やせば解消できそうであるが、冗長に思えるため、セオリーが知りたい。
    simd::float3 colors[POLYGON_INDICES] = {
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f},
    };
    uint16_t indices[DIV_POLYGON_INDICES] = {
        0, 1, 2,
        2, 3, 0,
    };

    const size_t positionDataSize = sizeof(simd::float3) * POLYGON_INDICES;
    const size_t colorDataSize = sizeof(simd::float3) * DIV_POLYGON_INDICES;
    const size_t indexDataSize = sizeof(simd::float3) * POLYGON_INDICES;

    MTL::Buffer* pVertexPositionsBuffer = _pDevice->newBuffer(positionDataSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer* pVertexColorsBuffer = _pDevice->newBuffer(colorDataSize, MTL::ResourceStorageModeManaged);
    MTL::Buffer* pIndicesBuffer = _pDevice->newBuffer(indexDataSize, MTL::ResourceStorageModeManaged);

    _pVertexPositionsBuffer = pVertexPositionsBuffer;
    _pVertexColorsBuffer = pVertexColorsBuffer;
    _pIndicesBuffer = pIndicesBuffer;

    memcpy(_pVertexPositionsBuffer->contents(), positions, positionDataSize);
    memcpy(_pVertexColorsBuffer->contents(), colors, colorDataSize);
    memcpy(_pIndicesBuffer->contents(), indices, indexDataSize);

    _pVertexPositionsBuffer->didModifyRange(NS::Range::Make(0, _pVertexPositionsBuffer->length()));
    _pVertexColorsBuffer->didModifyRange(NS::Range::Make(0, _pVertexColorsBuffer->length()));
    _pIndicesBuffer->didModifyRange(NS::Range::Make(0, _pIndicesBuffer->length()));
}

void Renderer::draw(MTK::View* pView) {
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    MTL::CommandBuffer* pCmd = _pCommandQueue->commandBuffer();
    MTL::RenderPassDescriptor* pRpd = pView->currentRenderPassDescriptor();
    MTL::RenderCommandEncoder* pEnc = pCmd->renderCommandEncoder(pRpd);

    pEnc->setRenderPipelineState(_pPSO);
    pEnc->setVertexBuffer(_pVertexPositionsBuffer, 0, 0);
    pEnc->setVertexBuffer(_pVertexColorsBuffer, 0, 1);
    pEnc->drawIndexedPrimitives(
        MTL::PrimitiveType::PrimitiveTypeTriangle,
        6,
        MTL::IndexType::IndexTypeUInt16,
        _pIndicesBuffer,
        0,
        1);

    pEnc->endEncoding();
    pCmd->presentDrawable(pView->currentDrawable());
    pCmd->commit();

    pPool->release();
}

#pragma endregion Renderer }
